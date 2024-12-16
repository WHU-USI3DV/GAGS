#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
import cv2
import matplotlib.pyplot as plt

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_map(network_output, gt):
    return torch.abs((network_output - gt)).mean(dim=0)

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()

def Scale_balance_loss(loss_map,seg_map,mask,scale_select_idx=1, mix_seg=False):
    '''
    scale_select_idx: 1,2,3 -> s,m,l default:1
    '''
    if mix_seg: # use mix seg of (H,W) instead of (3,H,W)
        mask_cur=seg_map!=-1
        exist_idx=torch.unique(seg_map[mask_cur]).to(torch.int)
    else:
        seg_map=F.interpolate(seg_map.unsqueeze(0),size=(mask.shape[0],mask.shape[1]),mode='nearest').squeeze(0)
        mask_cur=seg_map[scale_select_idx]!=-1
        exist_idx=torch.unique(seg_map[scale_select_idx][mask_cur]).to(torch.int)
    # print('min_idx,max_idx:',min_idx,max_idx,type(min_idx),type(max_idx))
    # print('seg_map:',seg_map.shape,seg_map.dtype)
    # print('loss_map:',loss_map.shape,loss_map.dtype)
    loss_list=[]
    for idx in exist_idx: # 逐区域计算mean loss
        if mix_seg:
            loss_i = loss_map[seg_map==idx.item()]
        else:
            loss_i = loss_map[seg_map[scale_select_idx]==idx.item()]
            
        if loss_i.numel() != 0:
            loss_list.append(torch.mean(loss_i))
            
    loss = torch.mean(torch.stack(loss_list))
    return loss

def scale_regulation_loss(scale_map):
    eps=1e-6
    c,h,w=scale_map.shape
    scale_map=scale_map.permute(1,2,0).reshape(-1,c) # H*W,3
    loss = -scale_map * torch.log(scale_map+eps)
    # loss=F.cross_entropy(scale_map, scale_map)
    loss=torch.mean(loss)
    return loss

def scale_regulation_loss_focal(scale_map):
    eps=1e-6
    c,h,w=scale_map.shape
    scale_map=scale_map.permute(1,2,0).reshape(-1,c) # H*W,3
    loss = -torch.log(scale_map+eps)*scale_map*(1-scale_map)**2 # H*W,3
    loss=torch.mean(loss)
    return loss

def scale_regulation_loss_s(scale_map):
    c,h,w=scale_map.shape
    scale_map=scale_map.permute(1,2,0).reshape(-1,c) # H*W,3
    # target=torch.zeros(h*w).to(torch.long).cuda() # target: [s,m,l]=[1,0,0]
    target=torch.tensor([1.0,0.5,0.0]).unsqueeze(0).repeat(h*w,1).to(torch.float32).cuda()
    loss=F.cross_entropy(scale_map, target)
    return loss

def scale_regulation_loss_var(scale_map):   
    s_value=0
    m_value=0.5
    l_value=1
    c,h,w=scale_map.shape
    scale_map=scale_map.permute(1,2,0).reshape(-1,c) # H*W,3

    avg_value=s_value*scale_map[:,0]+m_value*scale_map[:,1]+l_value*scale_map[:,2] # H*W
    loss=scale_map[:,0]*(s_value-avg_value)**2+scale_map[:,1]*(m_value-avg_value)**2+scale_map[:,2]*(l_value-avg_value)**2 # H*W
    loss=torch.mean(loss)
    return loss

def scale_regulation_loss_var2(scale_map):  
    c,h,w=scale_map.shape
    scale_map=scale_map.permute(1,2,0).reshape(-1,c) # H*W,3
    avg_scale=torch.mean(scale_map,dim=1).unsqueeze(1) # H*W, 1
    loss=torch.mean((scale_map-avg_scale)**2)
    return -loss

def scale_region_regulation_loss(scale_map,seg_map,scale_bal_idx=1, mix_seg=False):
    '''
    scale_bal_idx: 1,2,3 -> s,m,l
    '''
    c_scale,h_scale,w_scale=scale_map.shape
    if mix_seg:
        mask_cur=seg_map!=-1
        exist_idx=torch.unique(seg_map[mask_cur]).to(torch.int)
    else:
        seg_map=F.interpolate(seg_map.unsqueeze(0),size=(h_scale, w_scale),mode='nearest').squeeze(0)
        mask_cur=seg_map[scale_bal_idx]!=-1
        exist_idx=torch.unique(seg_map[scale_bal_idx][mask_cur]).to(torch.int)
    
    var_list=[]
    for idx in exist_idx: 
        if mix_seg:
            idx_cur = seg_map==idx.item()
        else:
            idx_cur = seg_map[scale_bal_idx]==idx.item() 
        # print('idx:',idx.item(),' pixels:',torch.sum(idx_cur).item())
        num_cur=torch.sum(idx_cur).item()
        # print('num_cur:',num_cur)
        if num_cur==0 or num_cur==1:
            continue
        idx_cur=idx_cur.unsqueeze(0).repeat(c_scale,1,1) # 3,h,w
        pix_cur=scale_map[idx_cur].reshape(c_scale,-1) # 3,n_pixel
        var_cur=torch.var(pix_cur,dim=1) # 3
        if torch.any(torch.isnan(var_cur)):
            print('var_cur:',var_cur)
            continue
        var_list.append(num_cur * var_cur.mean())
    # loss=torch.mean(torch.stack(var_list))
    loss = torch.sum(torch.stack(var_list)) / (h_scale*w_scale)
    return loss

def get_trained_seg(seg_map, scale_map):
    # print('seg_map:',seg_map.shape," scale_map:",scale_map.shape)
    def mean_smoothing(img, kernel_size=5):
        kernel = torch.ones(3, 1, kernel_size, kernel_size).to(img.device) / kernel_size**2
        return F.conv2d(img.unsqueeze(0), kernel, padding=kernel_size//2, groups=3).squeeze(0)
    seg_map=seg_map[1:,:,:] # 3,H,W
    scale_map=mean_smoothing(scale_map) # 3,H,W
    # seg_map=seg_map[1:,:,:]
    max_scale=torch.argmax(scale_map,dim=0) # H,W
    one_hot_scale=F.one_hot(max_scale,num_classes=3).permute(2,0,1) # 3,H,W
    # 可视化one_hot_scale
    # plt.figure()
    # plt.imshow(255*one_hot_scale.permute(1,2,0).cpu().numpy())
    # plt.show()
    # plt.close()
    seg_map_trained=torch.sum(seg_map*one_hot_scale,dim=0) # H,W
    return seg_map_trained
    
    
    
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

###
def tv_loss(feature_map):
    """
    Input:
    - feature_map: (C, H, W)
    Return:
    - total variation loss
    """
    tv_loss = ((feature_map[:, :, :-1] - feature_map[:, :, 1:])**2).sum() + ((feature_map[:, :-1, :] - feature_map[:, 1:, :])**2).sum()

    return tv_loss


def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_pixels = np.prod(y_true.shape)
    return correct_predictions / total_pixels
    
    
def calculate_iou(y_true, y_pred, num_classes):
    iou = []
    for i in range(num_classes):
        true_labels = y_true == i
        predicted_labels = y_pred == i
        intersection = np.logical_and(true_labels, predicted_labels)
        union = np.logical_or(true_labels, predicted_labels)
        iou_score = np.sum(intersection) / np.sum(union)
        iou.append(iou_score)
    return np.nanmean(iou)  

def calculate_avg_gradient(image):
    # 将图像转换为灰度图像（如果它不是灰度图像）
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    average_gradient_magnitude = np.mean(gradient_magnitude)
    
    return average_gradient_magnitude