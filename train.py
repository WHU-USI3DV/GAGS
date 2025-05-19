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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, l1_loss_map, Scale_balance_loss, scale_regulation_loss, scale_region_regulation_loss, get_trained_seg
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn.functional as F
from models.networks import CNN_decoder, CNN_scale_decoder
from scene.dataset_readers import read_sam_clip_feature
from segment_anything import sam_model_registry
from preprocess import OpenCLIPNetworkConfig, OpenCLIPNetwork


def create_scale_map(single_scale, feature_map_shape):
    scale_values = {
        "s": [1, 0, 0],
        "m": [0, 1, 0],
        "l": [0, 0, 1],
        "mix": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }
    assert single_scale in scale_values, "Invalid scale value"
    scale_map = torch.tensor(scale_values[single_scale], dtype=torch.float32, device='cuda')
    return scale_map.unsqueeze(-1).unsqueeze(-1).repeat(1, feature_map_shape[1], feature_map_shape[2])


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, scale_balance_iteration, scale_regulation_iteration, render_novel_view_iteration, novel_view_interval, feature_mode, single_scale):
    device0='cuda'
    device1='cpu'
        
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    # 2D semantic feature map CNN decoder
    viewpoint_stack = scene.getTrainCameras()
    camnum_orig=len(viewpoint_stack)
    viewpoint_cam0 = viewpoint_stack[0] 
    feature_out_dim = viewpoint_cam0.img_embed.shape[1]
    render_h,render_w = viewpoint_cam0.image_height, viewpoint_cam0.image_width
    print("render img with H,W:",render_h,",",render_w)
    
    # feature decoding 
    feature_in_dim = int(feature_out_dim / 32)
    if dataset.speedup:
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder = cnn_decoder.to(device0)
        cnn_decoder_optimizer = torch.optim.Adam(cnn_decoder.parameters(), lr=0.0001)
    # scale decoding
    cnn_scale_decoder = CNN_scale_decoder(feature_in_dim, 3)
    cnn_scale_decoder = cnn_scale_decoder.to(device0)
    cnn_scale_decoder_optimizer = torch.optim.Adam(cnn_scale_decoder.parameters(), lr=0.0001)
        
    original_h, original_w = viewpoint_cam0.semantic_feature_height, viewpoint_cam0.semantic_feature_width
    
    gaussians.training_setup(opt)
    if checkpoint: # continue from checkpoint
        (model_params, first_iter) = torch.load(checkpoint)
        if len(model_params) == 12 and feature_mode: 
            first_iter = 0
        else: # feature field
            # load feature decoder ckpt
            cnn_decoder_ckpt=torch.load(os.path.join(dataset.model_path, "decoder_chkpnt" + str(first_iter) + ".pth"))
            cnn_decoder.load_state_dict(cnn_decoder_ckpt['module_state_dict'])
            cnn_decoder_optimizer.load_state_dict(cnn_decoder_ckpt['optimizer_state_dict'])
            # load cscale decoder ckpt
            cnn_scale_decoder_ckpt=torch.load(os.path.join(dataset.model_path, "scale_decoder_chkpnt" + str(first_iter) + ".pth"))
            cnn_scale_decoder.load_state_dict(cnn_scale_decoder_ckpt['module_state_dict'])
            cnn_scale_decoder_optimizer.load_state_dict(cnn_scale_decoder_ckpt['optimizer_state_dict'])
        gaussians.restore(model_params, opt)
        print("number of gaussians",gaussians._xyz.shape)
    
    # set other parameters    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True) 
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
                
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()) #将net_image处理并通过memoryview提供一个直接访问这些数据的接口 clamp()将net_image的值截断至[0,1]
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy() 
        select_idx = randint(0, len(viewpoint_stack)-1) 
        viewpoint_cam = viewpoint_stack.pop(select_idx) 

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, feature_mode=feature_mode) 
        feature_map, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # 固定scale 
        if single_scale:
            scale_map = create_scale_map(single_scale, feature_map.shape)
        else:
            scale_map=cnn_scale_decoder(feature_map.detach()) # 3,H,W
        
        # Loss
        seg_map_trained = get_trained_seg(viewpoint_cam.seg_map.to(device0), scale_map)
        feature_reionvar_loss = scale_region_regulation_loss(feature_map, seg_map_trained, mix_seg=True) # feature 区域分布数据 方差
        # scale_reionvar_loss = scale_region_regulation_loss(scale_map, seg_map_trained, mix_seg=True) # scale 区域分布数据 方差
        
        scale_CE_loss = scale_regulation_loss(scale_map) 
        
        if dataset.speedup: 
            feature_map = cnn_decoder(feature_map) 

        if iteration < scale_balance_iteration: # L_distill
            gt_feature_map, seg_mask = read_sam_clip_feature(viewpoint_cam.img_embed.to(device0), viewpoint_cam.seg_map.to(device0), scale_map)
            Ll1_feature = l1_loss(feature_map * seg_mask, gt_feature_map * seg_mask)
        else: # L_r-distill
            gt_feature_map, seg_mask = read_sam_clip_feature(viewpoint_cam.img_embed.to(device0), viewpoint_cam.seg_map.to(device0), scale_map)
            Ll1_feature = l1_loss_map(feature_map * seg_mask, gt_feature_map * seg_mask) 
            Ll1_feature = Scale_balance_loss(Ll1_feature, seg_map_trained.to(device0), seg_mask.squeeze(0).to(device0), mix_seg=True)
            
        if iteration < scale_regulation_iteration:
            loss = 1.0 * Ll1_feature + 0.001 * scale_CE_loss 
        else:
            loss = 1.0 * Ll1_feature + 0.002 * scale_CE_loss + 0.1 * feature_reionvar_loss 
    
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if iteration % 500 == 0:
                # visualize scale_map every 500 iterations
                training_report(tb_writer, iteration, Ll1_feature, feature_reionvar_loss, 
                                torch.mean(scale_map[0]),torch.mean(scale_map[1]),torch.mean(scale_map[2]), 
                                loss, l1_loss, iter_start.elapsed_time(iter_end), 
                                testing_iterations, scene, render, (pipe, background), scale_map) 
            else:
                # regular log recording
                training_report(tb_writer, iteration, Ll1_feature, feature_reionvar_loss, 
                                torch.mean(scale_map[0]),torch.mean(scale_map[1]),torch.mean(scale_map[2]), 
                                loss, l1_loss, iter_start.elapsed_time(iter_end), 
                                testing_iterations, scene, render, (pipe, background))

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Max memory used: {mem:.2f} GB")
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and not feature_mode:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, feature_map.shape[2], feature_map.shape[1])
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold) # 增删高斯
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity() 
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if dataset.speedup:
                    cnn_decoder_optimizer.step()
                    cnn_decoder_optimizer.zero_grad(set_to_none = True)
                cnn_scale_decoder_optimizer.step()
                cnn_scale_decoder_optimizer.zero_grad(set_to_none = True)
                
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                print("\n[ITER {}] Saving feature decoder ckpt".format(iteration))
                if dataset.speedup:
                    torch.save({'module_state_dict':cnn_decoder.state_dict(), 
                                'optimizer_state_dict':cnn_decoder_optimizer.state_dict()
                                }, 
                               scene.model_path + "/decoder_chkpnt" + str(iteration) + ".pth")
                torch.save({'module_state_dict':cnn_scale_decoder.state_dict(), 
                            'optimizer_state_dict':cnn_scale_decoder_optimizer.state_dict()
                            }, 
                           scene.model_path + "/scale_decoder_chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1_feature, feature_reionvar_loss, scale_s, sclae_m, scale_l, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, scale_map=None):
    if tb_writer:
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss_feature', Ll1_feature.item(), iteration) 
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/feature_reionvar_loss', feature_reionvar_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('scale_patchs/subpart',scale_s.item(), iteration)
        tb_writer.add_scalar('scale_patchs/part',sclae_m.item(), iteration)
        tb_writer.add_scalar('scale_patchs/whole',scale_l.item(), iteration)
        
        # visualize scale_map
        if scale_map is not None:

            # add to tensorboard
            tb_writer.add_image('scale_map_rgb', scale_map, iteration, dataformats='CHW')
            
            # add 3 separate heatmaps for s, m, l granularity
            tb_writer.add_image('scale_map/s_scale', scale_map[0].unsqueeze(0), iteration, dataformats='CHW')
            tb_writer.add_image('scale_map/m_scale', scale_map[1].unsqueeze(0), iteration, dataformats='CHW')
            tb_writer.add_image('scale_map/l_scale', scale_map[2].unsqueeze(0), iteration, dataformats='CHW')

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser) 
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--scale_balance_iteration', type=int, default=1)
    parser.add_argument('--scale_regulation_iteration', type=int, default=15001)
    parser.add_argument('--render_novel_view_iteration',type=int, default=99999)
    parser.add_argument('--novel_view_interval',type=int,default=150)
    parser.add_argument('--feature_mode', action='store_true', help='use feature replace RGB')
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument("--novel_view", action='store_true')
    parser.add_argument("--single_scale",type=str, choices=['s', 'm', 'l', 'mix'], default = None) # s | m | l
    args = parser.parse_args(sys.argv[1:]) 
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Initialize SAM & CLIP model
    if args.novel_view:
        CLIP_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
        sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt_path).to('cuda')
    else:
        CLIP_model = None
        sam = None
        
    # empty cache
    torch.cuda.empty_cache()
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, 
             args.debug_from, args.scale_balance_iteration, args.scale_regulation_iteration,args.render_novel_view_iteration,args.novel_view_interval,args.feature_mode,args.single_scale)

    # All done
    print("\nTraining complete.")
