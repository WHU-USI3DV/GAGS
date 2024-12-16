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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render 
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import sklearn
import sklearn.decomposition
import numpy as np
from PIL import Image
import torch.nn as nn
from models.networks import CNN_decoder, CNN_scale_decoder
from scene.dataset_readers import read_sam_clip_feature
import matplotlib.pyplot as plt
import glob

def feature_visualize_saving(feature):
    fmap = feature[None, :, :, :] # torch.Size([1, 512, h, w])
    fmap = nn.functional.normalize(fmap, dim=1) # channel维度上做归一化
    pca = sklearn.decomposition.PCA(3, random_state=42) # PCA降维到3维
    f_samples = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])[::3].cpu().numpy() # 1, h, w, 512 -> h*w/3, 512
    transformed = pca.fit_transform(f_samples)
    feature_pca_mean = torch.tensor(f_samples.mean(0)).float().cuda()
    feature_pca_components = torch.tensor(pca.components_).float().cuda()
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    del f_samples
    vis_feature = (fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1]) - feature_pca_mean[None, :]) @ feature_pca_components.T
    vis_feature = (vis_feature - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    vis_feature = vis_feature.clamp(0.0, 1.0).float().reshape((fmap.shape[2], fmap.shape[3], 3)).cpu() # h, w, 3
    return vis_feature

def scale_visualize_saving(scale_map):
    max_scale=torch.argmax(scale_map,dim=0) # 0,1,2
    max_scale=max_scale/2
    return max_scale

def process_scale_map(scale_map):
    scale_maps = [torch.zeros_like(scale_map).cuda() for _ in range(3)]
    for i, sm in enumerate(scale_maps):
        sm[i] = 1
    return scale_maps

def process_feature_map(view, scale_map):
    gt_feature_maps = []
    for sm in process_scale_map(scale_map):
        gt_feature_map, mask = read_sam_clip_feature(view.img_embed.cuda(), view.seg_map.cuda(), sm.cuda(), max_mode=True)
        gt_feature_maps.append(gt_feature_map * mask)
    return gt_feature_maps

def render_set(model_path, source_path, name, iteration, views, gaussians, pipeline, background, speedup, feature_mode, feature_npy, render_mode):
    image_gt_list = glob.glob(os.path.join(source_path, 'images', '*.*'))
    image_gt_list.sort()
    orig_img_width, orig_img_height = Image.open(image_gt_list[0]).size
    print("gt image size:", orig_img_height, "," , orig_img_width)
    
    device0 = 'cuda'
    device1 = 'cpu'
    
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map")
    gt_feature_map_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_feature_map")
    scale_map_path=os.path.join(model_path, name, "ours_{}".format(iteration), "scale_map")
    feature_map_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "feature_map_npy")
    #encoder_ckpt_path = os.path.join(model_path, "encoder_chkpnt{}.pth".format(iteration))
    decoder_ckpt_path = os.path.join(model_path, "decoder_chkpnt{}.pth".format(iteration))
    scale_decoder_ckpt_path=os.path.join(model_path, "scale_decoder_chkpnt{}.pth".format(iteration))
    depth_data_path=os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
    depth_visual_path=os.path.join(model_path, name, "ours_{}".format(iteration), "depths_visual")
    
    if feature_mode and feature_npy==False:
        feature_out_dim=views[0].img_embed.shape[1] # 512 / 768
        feature_in_dim = int(feature_out_dim/32)
        if speedup:
            cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
            cnn_decoder.load_state_dict(torch.load(decoder_ckpt_path)['module_state_dict'])

        cnn_scale_decoder=CNN_scale_decoder(feature_in_dim, 3)
        cnn_scale_decoder.load_state_dict(torch.load(scale_decoder_ckpt_path)['module_state_dict'])
    
    if feature_npy:
        makedirs(feature_map_npy_path, exist_ok=True)
    elif feature_mode==False:
        makedirs(render_path, exist_ok=True)
        # makedirs(gts_path, exist_ok=True)
        if render_mode == "RGB+ED":
            makedirs(depth_data_path, exist_ok=True)
            makedirs(depth_visual_path, exist_ok=True)
    else:
        makedirs(feature_map_path, exist_ok=True)
        makedirs(gt_feature_map_path, exist_ok=True)
        makedirs(scale_map_path, exist_ok=True)
        
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        if feature_npy or feature_mode==False: # change to full resolution
            view.image_width = orig_img_width
            view.image_height = orig_img_height
        
        render_pkg = render(view, gaussians, pipeline, background, feature_mode=feature_mode, render_mode=render_mode) 
        feature_map = render_pkg["render"] # 16,731,989
        
        if feature_npy: # only save *.npy feature map file
            np.save(os.path.join(feature_map_npy_path, '{0:05d}'.format(idx) + ".npy"), feature_map.permute(1,2,0).cpu().numpy()) # H, W, 16
        
        elif feature_mode==False: # rendering RGB(&depth)
            c,h,w=feature_map.shape
            # gt = view.original_image[0:3, :, :]
            rendering_RGB=feature_map[:3, :, :]   # 3，H，W
            
            if c == 4: # including depth
                rendering_depth=feature_map[3, :, :]  # H，W
                rendering_depth=rendering_depth.cpu().numpy()
                # save depth
                np.save(os.path.join(depth_data_path, view.image_name + "_depth.npy"),rendering_depth)
                
                # save visualized depth
                plt.figure("Image",(w/100.0,h/100.0),dpi=100)     # 图像窗口名称
                plt.imshow(rendering_depth, cmap='viridis')
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                plt.savefig(os.path.join(depth_visual_path, view.image_name + "_depth.png"), bbox_inches='tight', pad_inches=0)   
                plt.close()
            
            # save render RGB & gt RGB (use image name)
            torchvision.utils.save_image(rendering_RGB, os.path.join(render_path, view.image_name + "_RGB.png"))
            # torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
            
        else: # rendering feature map
            scale_map=cnn_scale_decoder(feature_map.detach()) # 3, 731, 989

            torchvision.utils.save_image(scale_map, os.path.join(scale_map_path, '{0:05d}'.format(idx) + ".png"))
            scale_map_save=scale_visualize_saving(scale_map)
            torchvision.utils.save_image(scale_map_save, os.path.join(scale_map_path, '{0:05d}'.format(idx) + "_class.png"))
            
            gt_feature_map, mask = read_sam_clip_feature(view.img_embed.cuda(),view.seg_map.cuda(),scale_map.cuda(),max_mode=True) # 512, 47, 64
            gt_feature_map = gt_feature_map * mask
            
            gt_feature_map_s, gt_feature_map_m, gt_feature_map_l = process_feature_map(view, scale_map)
            
            # torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png")) 
            # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(mask.to(torch.float32), os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_mask.png"))
            if speedup:
                feature_map = cnn_decoder(feature_map) # C, H, W                   
            
            feature_map_vis = feature_visualize_saving(feature_map)
            Image.fromarray((feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))
            gt_feature_map_vis = feature_visualize_saving(gt_feature_map)
            Image.fromarray((gt_feature_map_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis.png"))

            gt_feature_map_s_vis=feature_visualize_saving(gt_feature_map_s)
            Image.fromarray((gt_feature_map_s_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis_s.png"))
            gt_feature_map_m_vis=feature_visualize_saving(gt_feature_map_m)
            Image.fromarray((gt_feature_map_m_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis_m.png"))
            gt_feature_map_l_vis=feature_visualize_saving(gt_feature_map_l)
            Image.fromarray((gt_feature_map_l_vis.cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(gt_feature_map_path, '{0:05d}'.format(idx) + "_feature_vis_l.png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, feature_mode: bool, feature_npy: bool, render_mode: str): ###
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False) # 从.ply文件中加载场景信息

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        dataset.speedup = getattr(dataset, 'speedup', False)

        if not skip_train:
             render_set(dataset.model_path, dataset.source_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.speedup, feature_mode, feature_npy, render_mode)

        if not skip_test:
             render_set(dataset.model_path, dataset.source_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.speedup, feature_mode, feature_npy, render_mode)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--feature_mode', action='store_true', help='use feature replace RGB')
    parser.add_argument("--feature_npy", action='store_true', help='store 16-dim feature map in npy')
    parser.add_argument("--render_mode", default="RGB", type=str) # RGB+ED
    args = get_combined_args(parser) #从命令行获取参数，并根据得到的model_path解析该目录下cfg_args文件，获取训练参数
    print("Rendering " + args.model_path)

    assert not (args.feature_mode and args.render_mode == "RGB+ED"), "Feature mode does not support depth rendering"
    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(args)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.feature_mode, args.feature_npy, args.render_mode) ###cnn