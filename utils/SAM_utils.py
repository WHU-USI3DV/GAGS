import os
import random
import argparse
import sys
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Any, Dict, Generator, ItemsView, List, Tuple
import math
from itertools import product
import math

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep
def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def sam_encoder(image):
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps   
def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx
def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new 

def mask2segmap(masks, image):
    seg_img_list = []
    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    for i in range(len(masks)):
        mask = masks[i]
        seg_img = get_seg_img(mask, image)
        pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
        seg_img_list.append(pad_seg_img)

        seg_map[masks[i]['segmentation']] = i
    seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
    seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

    return seg_imgs, seg_map

def random_colormap(num_colors):
    colors = np.random.rand(num_colors, 3) 
    colors = np.clip(colors, 0, 1)  
    cmap = mcolors.ListedColormap(colors)
    return cmap

def build_point_grid(n_per_side: int) -> np.ndarray: 
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points # n*n,2

def build_all_layer_point_grids( 
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer # list(array(n*n,2))

def generate_crop_boxes( 
    im_size: Tuple[int, ...], n_layers: int, overlap_ratio: float
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates a list of crop boxes of different sizes. Each layer
    has (2**i)**2 boxes for the ith layer.
    """
    crop_boxes, layer_idxs = [], []
    im_h, im_w = im_size
    short_side = min(im_h, im_w)

    # Original image
    crop_boxes.append([0, 0, im_w, im_h]) 
    layer_idxs.append(0) 

    def crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_w = crop_len(im_w, n_crops_per_side, overlap)
        crop_h = crop_len(im_h, n_crops_per_side, overlap)

        crop_box_x0 = [int((crop_w - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_h - overlap) * i) for i in range(n_crops_per_side)]

        # Crops in XYWH format
        for x0, y0 in product(crop_box_x0, crop_box_y0):
            box = [x0, y0, min(x0 + crop_w, im_w), min(y0 + crop_h, im_h)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs

def build_depth_point_grid(n_per_side: int, depth_map: torch.tensor) -> np.ndarray:
    """Generates point grid based on depth maps."""
    sample_points=[]
    sample_boxs=[]
    h,w=depth_map.shape
    crop_x0=np.linspace(0,w-1,n_per_side+1)[:-1].astype(np.int32)
    crop_w=int(w/len(crop_x0))
    crop_y0=np.linspace(0,h-1,n_per_side+1)[:-1].astype(np.int32)
    crop_h=int(h/len(crop_y0))
    # print(crop_x0,crop_y0,crop_w,crop_h)
    # print(depth_map.shape)
    for x0, y0 in product(crop_x0, crop_y0):
        mean_depth=torch.mean(depth_map[y0:min(y0 + crop_h, h),x0:min(x0 + crop_w, w)])
        sample_num=int(mean_depth)
        if sample_num>20:
            sample_num=20
        elif sample_num<1:
            sample_num=1
        # print('mean_depth(',x0,y0,')(',x0+crop_w,y0+crop_h,")=",mean_depth)
        offset_x = crop_w / (2 * sample_num)
        offset_y = crop_h / (2 * sample_num)
        points_axis_x = np.linspace(x0 + offset_x, x0 + crop_w - offset_x, sample_num)
        points_axis_y = np.linspace(y0 + offset_y, y0 + crop_h - offset_y, sample_num)
        points_x = np.tile(points_axis_x[None, :], (sample_num, 1))
        points_y = np.tile(points_axis_y[:, None], (1, sample_num))
        points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2) # n*n,2
        sample_points.append(points)
        sample_boxs.append(np.array([x0/w, y0/h, (x0+crop_w)/w, (y0+crop_h)/h]))
        
    sample_points_concat=np.concatenate(sample_points, axis=0) # N_points,2
    sample_boxs_concat=np.stack(sample_boxs, axis=0) # N_boxs,4
    points_scale = np.array(depth_map.shape)[None, ::-1] # 1,2
    # print('points_scale=',points_scale)
    sample_points_concat=sample_points_concat/points_scale 
    # print(sample_points_concat.shape)
    return sample_points_concat,sample_boxs_concat

def build_all_layer_depth_point_grids( # Generate normalized grid points for each level
    n_per_side: int, n_layers: int, scale_per_layer: int, depth_map: torch.tensor
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    boxs_by_layer=[]
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points,box=build_depth_point_grid(n_points, depth_map)
        points_by_layer.append(points)
        boxs_by_layer.append(box)
    return points_by_layer,boxs_by_layer # list(array(n_points,2)) list(array(n_boxs,4))

def sample_based_mapping(sample_depth_crop, sample_num, crop_num = 10):
    h,w = sample_depth_crop.shape
    crops_axis_x = np.linspace(0, w-1, crop_num+1)[:-1].astype(np.int32)
    crops_axis_y = np.linspace(0, h-1, crop_num+1)[:-1].astype(np.int32)
    # print('crops_axis_x:',len(crops_axis_x))
    allcrops_x=np.tile(crops_axis_x[None, :], (crop_num, 1)).reshape(crop_num*crop_num)
    allcrops_y=np.tile(crops_axis_y[:, None], (1, crop_num)).reshape(crop_num*crop_num)
    num_list=[]
    idx_list=[i for i in range(crop_num**2)]
    for i in range(allcrops_x.shape[0]):
        sample_depth = sample_depth_crop[allcrops_y[i]:min(h-1,allcrops_y[i]+h//crop_num), allcrops_x[i]:min(w-1,allcrops_x[i]+w//crop_num)]
        num_non_zero = torch.sum(sample_depth != 0)
        num_list.append(num_non_zero)
    if np.all(np.array(num_list) == 0): # If all sub-crops have no non-zero points, set the sampling probability to uniform distribution
        num_list = [1] * len(num_list)
    
    # random sample
    weights = num_list/np.sum(num_list)
    sample_num_list = random.choices(idx_list, weights, k=sample_num)

    point_list=[]
    for i, sample_cur in enumerate(sample_num_list):
        sample_x0, sample_y0 = allcrops_x[sample_cur], allcrops_y[sample_cur]
        sample_x1, sample_y1 = min(w-1,allcrops_x[sample_cur]+w//crop_num),  min(h-1,allcrops_y[sample_cur]+h//crop_num)
        point_list.append([random.randint(sample_x0, sample_x1), random.randint(sample_y0, sample_y1)])
    return point_list

def build_mindepth_point_grid(n_per_side: int, depth_map: torch.tensor, depth_sample: torch.tensor, nsample_min_distance: int
                              ) -> np.ndarray:
    """Generates point grid based on depth maps and depth samples."""
    sample_points=[]
    sample_boxs=[]
    h,w=depth_map.shape
    crop_x0=np.linspace(0,w-1,n_per_side+1)[:-1].astype(np.int32)
    crop_w=int(w/len(crop_x0))
    crop_y0=np.linspace(0,h-1,n_per_side+1)[:-1].astype(np.int32)
    crop_h=int(h/len(crop_y0))

    for x0, y0 in product(crop_x0, crop_y0):
        mean_depth=torch.mean(depth_map[y0:min(y0 + crop_h, h),x0:min(x0 + crop_w, w)])
        sample_depth_crop=depth_sample[y0:min(y0 + crop_h, h),x0:min(x0 + crop_w, w)]
        mean_sample_depth=torch.mean(sample_depth_crop[sample_depth_crop!=0])
        if (mean_depth/mean_sample_depth)<1 or (mean_depth/mean_sample_depth).isnan():
            sample_num=1
        else:
            sample_num=int((mean_depth/mean_sample_depth) * nsample_min_distance)
        sample_num = max(1, min(sample_num, 20))
        points_from_sample = sample_based_mapping(sample_depth_crop, sample_num**2)
        points_from_sample = [[x0+point[0], y0+point[1]] for point in points_from_sample]
        points_from_sample = np.array(points_from_sample) # n*n,2
        sample_points.append(points_from_sample)
        sample_boxs.append(np.array([x0/w, y0/h, (x0+crop_w)/w, (y0+crop_h)/h]))
        
    sample_points_concat=np.concatenate(sample_points, axis=0) # N_points,2
    sample_boxs_concat=np.stack(sample_boxs, axis=0) # N_boxs,4
    points_scale = np.array(depth_map.shape)[None, ::-1] # 1,2
    # print('points_scale=',points_scale)
    sample_points_concat=sample_points_concat/points_scale 
    # print(sample_points_concat.shape)
    return sample_points_concat,sample_boxs_concat

def build_all_layer_mindepth_point_grids( # Generate normalized grid points for each level
    n_per_side: int, n_layers: int, scale_per_layer: int, nsample_min_distance: int, depth_map: torch.tensor, depth_sample: torch.tensor
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    boxs_by_layer=[]
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points,box=build_mindepth_point_grid(n_points, depth_map, depth_sample, nsample_min_distance)
        points_by_layer.append(points)
        boxs_by_layer.append(box)
    return points_by_layer,boxs_by_layer # list(array(n_points,2)) list(array(n_boxs,4))

def project_from_sampled_pcd(pcd_pxl_mask, pcd_pxl_mapping, n_layers : int, h, w) -> List[np.ndarray]:
    """Project point to each imgs with mapping matrix."""
    points_by_layer = []
    for i in range(n_layers + 1):
        points=pcd_pxl_mapping[pcd_pxl_mask].astype(np.float32) # N_points,2
        points[:, 0]=points[:, 0]/h
        points[:, 1]=points[:, 1]/w
        points=np.stack((points[:, 1],points[:, 0]),axis=-1) # N_points,2
        # print('points:',points.shape, points[:20,:])
        points_by_layer.append(points)
    return points_by_layer # list(array(n_points,2))
    
def sample_from_pcd(pcd_depth, pcd_pxl_mask, sample_num):
    pcd_pxl_mask=torch.from_numpy(pcd_pxl_mask)
    point_ids = torch.unique(pcd_pxl_mask.nonzero(as_tuple=False)[:, 0]) # [N_points] 有对应2D pixel的3D points index
    pcd_depth=pcd_depth[point_ids]  # [N_valid_points]
    print('max_depth:',np.max(pcd_depth),'min_depth:',np.min(pcd_depth))
    weights=pcd_depth/np.sum(pcd_depth)
    sample_idx_list = random.choices(point_ids, weights, k=sample_num)
    unique_sample_idx_list = sorted(set(sample_idx_list))
    return unique_sample_idx_list

if __name__ == '__main__':
    
    '''
    testing all SAM propmt modes 
    '''
    
    parser = argparse.ArgumentParser(description='SAM vis')
    parser.add_argument('--out_path', type=str, default='SAM', help='output path')
    args = parser.parse_args(sys.argv[1:]) # 从命令行获取参数
    
    case_name='case_name'
    dataset_name ='ramen'
    iteration = 30000
    
    img_path=f'data/{dataset_name}/images' 
    depth_path=f'data/{dataset_name}/depths'
    depth_sample_path=f'output/{case_name}/train/ours_{iteration}/depths_sample'
    pcd_depth_sample_path=f'output/{case_name}/train/ours_{iteration}/pcd_depths_sample'
    save_path=os.path.join('SAM', case_name + "_16_2_" + args.out_path)
    os.makedirs(save_path,exist_ok=True)
    sam_ckpt_path='ckpts/sam_vit_h_4b8939.pth'
    pcd_depth=np.load(os.path.join(pcd_depth_sample_path,'pcd_depth.npy')) # N_points
    pcd_pxl_mask=np.load(os.path.join(pcd_depth_sample_path,'pcd_pxl_mask.npy')) # N_points, N_cameras
    pcd_pxl_mapping=np.load(os.path.join(pcd_depth_sample_path,'pcd_pxl_mapping.npy')) # N_points, N_cameras, 2

    # depth params
    pps_depth=16 
    nlayers_depth=0 
    nsample_min_distance=2 
    sample_num_pcd = round(0.005*pcd_depth.shape[0]) 
    
    # default params
    pps=32 
    nlayers=1
    
    # generic params
    scale_per_layer=1 
    overlap_ratio=512/1500 
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    img_name_list = os.listdir(img_path)
    img_name_list.sort()
    depth_name_path= os.listdir(depth_path)
    depth_name_path.sort()
    depth_sample_name_list=os.listdir(depth_sample_path)
    depth_sample_name_list.sort()
    assert len(img_name_list)==len(depth_name_path), print('img_name_list!=depth_name_path')
    img_fullname_list=[os.path.join(img_path,img_name_list[i]) for i in range(len(img_name_list))]
    depth_fullname_list=[os.path.join(depth_path,depth_name_path[i]) for i in range(len(depth_name_path))]
    depth_sample_fullname_list=[os.path.join(depth_sample_path,depth_sample_name_list[i]) for i in range(len(depth_sample_name_list))]
    
    # sample for pcd_min_depth mode
    sample_idx = sample_from_pcd(pcd_depth, pcd_pxl_mask, sample_num_pcd)
    print('pcd sample num:',len(sample_idx))
    
    for idx in range(len(img_fullname_list)):
        
        print('processing image',img_fullname_list[idx], '...')
        image = cv2.imread(img_fullname_list[idx]) # HWC BGR
        h,w=image.shape[:2]
        # image = torch.from_numpy(image)
        image = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_BGR2RGB)
        depth_image=torch.from_numpy(np.load(depth_fullname_list[idx]))
        depth_sample=torch.from_numpy(np.load(depth_sample_fullname_list[idx]))
        depth_mode=True
        min_depth_mode=True
        pcd_min_depth_mode=True
        
        for n in range(4):
            
            # print('depth_mode:',depth_mode)
            if depth_mode:
                depth_sample_points,depth_sample_boxs=build_all_layer_depth_point_grids(pps_depth,nlayers_depth,scale_per_layer,depth_image)
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=None, ### default
                    point_grids=depth_sample_points, ### depth based
                    pred_iou_thresh=0.7,
                    box_nms_thresh=0.7,
                    stability_score_thresh=0.85,
                    crop_n_layers=nlayers_depth,
                    crop_n_points_downscale_factor=scale_per_layer,
                    min_mask_region_area=100,
                    crop_overlap_ratio=overlap_ratio
                )
            elif min_depth_mode:
                depth_sample_points,depth_sample_boxs=build_all_layer_mindepth_point_grids(pps_depth,nlayers_depth,scale_per_layer,nsample_min_distance,depth_image,depth_sample)
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=None, ### default
                    point_grids=depth_sample_points, ### depth based
                    pred_iou_thresh=0.7,
                    box_nms_thresh=0.7,
                    stability_score_thresh=0.85,
                    crop_n_layers=nlayers_depth,
                    crop_n_points_downscale_factor=scale_per_layer,
                    min_mask_region_area=100,
                    crop_overlap_ratio=overlap_ratio
                )
            elif pcd_min_depth_mode:
                depth_sample_points = project_from_sampled_pcd(pcd_pxl_mask[sample_idx, idx].astype(bool), pcd_pxl_mapping[sample_idx, idx], nlayers_depth, h, w)   
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=None, ### default
                    point_grids=depth_sample_points, ### depth based
                    pred_iou_thresh=0.7,
                    box_nms_thresh=0.7,
                    stability_score_thresh=0.85,
                    crop_n_layers=nlayers_depth,
                    crop_n_points_downscale_factor=scale_per_layer,
                    min_mask_region_area=100,
                    crop_overlap_ratio=overlap_ratio
                )
            else:
                mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=pps, ### default
                    point_grids=None, ### depth based
                    pred_iou_thresh=0.7,
                    box_nms_thresh=0.7,
                    stability_score_thresh=0.85,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=scale_per_layer,
                    min_mask_region_area=100,
                    crop_overlap_ratio=overlap_ratio
                )
                
            # SAM segmentation
            # pre-compute masks
            masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
            
            # pre-compute postprocess
            masks_default, masks_s, masks_m, masks_l = \
                masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)

            # print("mask data",type(masks_default))
            seg_images, seg_maps = {}, {}
            _, seg_maps['default'] = mask2segmap(masks_default, image)
            if len(masks_s) != 0:
                _, seg_maps['s'] = mask2segmap(masks_s, image)
            if len(masks_m) != 0:
                _, seg_maps['m'] = mask2segmap(masks_m, image)
            if len(masks_l) != 0:
                _, seg_maps['l'] = mask2segmap(masks_l, image)
            
            # print("seg_maps",seg_maps['s'].shape,type(seg_maps['s']))
            # print("seg_images",seg_images['s'].shape,type(seg_images['s']))
            
            # visualization
            num_colors = 256 
            bounds = np.arange(num_colors+1)-1.5 # [-1.5 -0.5 0.5 ... 254.5]
            cmap = random_colormap(num_colors)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            plt.figure("seg_maps",(20,10),dpi=300)   
            plt.subplot(2, 2, 1)  
            plt.axis([0, w, h, 0]) # x0,x1,y1,y0
            plt.imshow(image)
            num_points=0
            
            if depth_mode:
                # visualization prompt & crop 
                for i in range(len(depth_sample_points)):
                    num_points+=depth_sample_points[i].shape[0]
                    plt.scatter(depth_sample_points[i][:,0]*w,depth_sample_points[i][:,1]*h,s=10,c='r')
                    # print('depth_sample_boxs[i].shape=',depth_sample_boxs[i].shape)
                    for j in range(depth_sample_boxs[i].shape[0]):
                        x0,y0,x1,y1=depth_sample_boxs[i][j,:]
                        plt.plot([x0*w,x1*w,x1*w,x0*w,x0*w],[y0*h,y0*h,y1*h,y1*h,y0*h],c='yellow')   
                
            elif min_depth_mode:
                # visualization prompt & crop 
                for i in range(len(depth_sample_points)):
                    num_points+=depth_sample_points[i].shape[0]
                    plt.scatter(depth_sample_points[i][:,0]*w,depth_sample_points[i][:,1]*h,s=10,c='r')
                    # print('depth_sample_boxs[i].shape=',depth_sample_boxs[i].shape)
                    for j in range(depth_sample_boxs[i].shape[0]):
                        x0,y0,x1,y1=depth_sample_boxs[i][j,:]
                        plt.plot([x0*w,x1*w,x1*w,x0*w,x0*w],[y0*h,y0*h,y1*h,y1*h,y0*h],c='yellow')   
            elif pcd_min_depth_mode:
                # visualization prompt
                for i in range(len(depth_sample_points)):
                    num_points+=depth_sample_points[i].shape[0]
                    plt.scatter(depth_sample_points[i][:,0]*w,depth_sample_points[i][:,1]*h,s=10,c='r')
                    
            else:
                point_grids = build_all_layer_point_grids(pps, nlayers, scale_per_layer) # list(np array(n*n,2))
                crop_boxs, layer_levels = generate_crop_boxes((h,w), nlayers, overlap_ratio) # list(list(x0,y0,x1,y1)) list(level)
                # visualization prompt & crop：default
                for i in range(len(crop_boxs)):
                    layer_level=layer_levels[i]
                    num_points+=point_grids[layer_level].shape[0]
                    x0,y0,x1,y1=crop_boxs[i]
                    # print("crop_boxs",i,": ",x0," ",y0," ",x1," ",y1)
                    plt.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],color=np.random.rand(3,))
                    box_h=y1-y0
                    box_w=x1-x0
                    color_map=["orange","red","purple","blue","black"]
                    points_x=point_grids[layer_level][:,0]*box_w+x0
                    points_y=point_grids[layer_level][:,1]*box_h+y0
                    plt.scatter(points_x, points_y, color=color_map[layer_level], s=10)

            plt.subplot(2, 2, 2)
            plt.imshow(seg_maps['s'], cmap=cmap,norm=norm)
            
            plt.subplot(2, 2, 3)
            plt.imshow(seg_maps['m'], cmap=cmap,norm=norm)
            
            plt.subplot(2, 2, 4)
            plt.imshow(seg_maps['l'], cmap=cmap,norm=norm)
            
            plt.tight_layout()
            
            img_name=img_fullname_list[idx].split('/')[-1].split('.')[0]
            if depth_mode:
                save_img_name='depth_'+img_name+'_'+str(pps_depth)+"_"+str(num_points)+'.png'
                depth_mode = not depth_mode
            elif min_depth_mode:
                save_img_name='mindepth_'+img_name+'_'+str(pps_depth)+"_"+str(num_points)+'.png'
                min_depth_mode = not min_depth_mode
            elif pcd_min_depth_mode:
                save_img_name='pcd_mindepth_'+img_name+'_'+str(pps_depth)+"_"+str(num_points)+'.png'
                pcd_min_depth_mode = not pcd_min_depth_mode
            else:
                save_img_name= img_name+'_'+str(pps)+"_"+str(num_points)+'.png'
                depth_mode = not depth_mode
                min_depth_mode = not min_depth_mode
                pcd_min_depth_mode = not pcd_min_depth_mode
                
            plt.savefig(os.path.join(save_path, save_img_name), dpi=300)
            print('successfully save at',save_img_name)
            # plt.show()
            plt.close()
            
        