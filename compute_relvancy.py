import torch
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import matplotlib
matplotlib.use('Agg')   # opencv-python has a crash with matplotlib(interactive backend),
                        # uninstall PyQt5 may solve this problem
import matplotlib.pyplot as plt
import numpy as np
from models.networks import CNN_decoder, CNN_scale_decoder
from scene.dataset_readers import read_sam_clip_feature

from utils.preprocess_utils import OpenCLIPNetwork, OpenCLIPNetworkConfig

from eval import colormaps
import glob
from eval.openclip_encoder import OpenCLIPNetwork as OpenCLIPNetwork_eval
import matplotlib.patches as patches
import matplotlib.colors as pltcolors
from plyfile import PlyData
from utils.pcd_utils import vis_pcd, create_novel_view, smooth_pcd_mask
from eval.utils import smooth_GPU, colormap_saving
from pathlib import Path

from utils.campath_generate_utils import generate_interpolated_path, simple_interpolation
from typing import Literal

C0 = 0.28209479177387814 
    
def show_result(image, save_path):
    plt.figure()
    plt.imshow(image)
    rect = patches.Rectangle((0, 0), image.shape[1]-1, image.shape[0]-1, linewidth=0, edgecolor='none', facecolor='white', alpha=0.3)
    plt.gca().add_patch(rect)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=200)
    plt.close()

def show_and_save_loss_map(mean_featuredim_loss, feature_map, GT_feature_map, output_path, iteration, idx):
    
    output_path_loss = output_path / 'train' / f"ours_{iteration}" / "loss_map"
    os.makedirs(output_path_loss, exist_ok=True)
    
    fig, axs = plt.subplots(1, 3, dpi=100, figsize=(20,5))
    color_dic={'vmin':0,'vmax':0.035}
    color_dic_01={'vmin':0,'vmax':1}
    fig1 = axs[0].imshow(GT_feature_map.cpu().detach().numpy(),**color_dic)
    axs[0].set_title('GT feature map value (avg. along dim)')
    fig.colorbar(fig1, ax=axs[0])
    
    fig2 = axs[1].imshow(feature_map.cpu().detach().numpy(),**color_dic)
    axs[1].set_title('feature map value (avg. along dim)')
    fig.colorbar(fig2, ax=axs[1])
    
    fig3 = axs[2].imshow(mean_featuredim_loss.cpu().detach().numpy(),cmap='gnuplot2',**color_dic_01)
    axs[2].set_title('feature L2 loss (avg. along dim)')
    fig.colorbar(fig3, ax=axs[2])
    
    # save
    plt.savefig(output_path_loss / f"cam_{idx:0>5}.png",dpi=200)
    plt.show()
    plt.close()
    
def activate_stream(sem_map, 
                    image, 
                    clip_model, 
                    output_path: Path,
                    idx: int = 0,
                    thresh : float = 0.5, 
                    colormap_options : colormaps.ColormapOptions = None):
    """
    Compute and visualize relevancy of text query with lerf-style and GAGS-style.
    Args:
        sem_map (_type_): Current full-dim semantic feature map. (1, H, W, 512)
        image (_type_): Current RGB image. (H, W, 3)
        clip_model (_type_): CLIP model used for relevancy computation.
        output_path (Path): Output path for saving results.
        idx (int): Current view index.
        thresh (float, optional): Threshold for mask generation. Defaults to 0.5.
        colormap_options: Settings for colormaps for visualization.
    """
    valid_map = clip_model.get_max_across(sem_map).squeeze(0)                # k,H,W
    n_prompt, _, _ = valid_map.shape

    output_path_relev = output_path / 'heatmap'
    output_path_compo = output_path / 'lerf_composited'
    output_path_lerf_compo = output_path / 'lerf_composited_whitebg'
    output_path_mask_compo = output_path / 'mask_composited'
    output_path_relev.mkdir(exist_ok=True, parents=True)
    output_path_compo.mkdir(exist_ok=True, parents=True)
    output_path_lerf_compo.mkdir(exist_ok=True, parents=True)
    output_path_mask_compo.mkdir(exist_ok=True, parents=True)
    
    # positive prompts
    for k in range(n_prompt):
        # smooth
        scale = 30
        kernel = np.ones((scale,scale)) / (scale**2)
        np_relev = valid_map[k].cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev, -1, kernel)
        avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
        valid_map[k] = 0.5 * (avg_filtered + valid_map[k])
        
        # (lerf/langsplat-style) heatmap(smoothed relvancy map)
        # rel_map -> norm[0,1] -> clip[0.5,1] -> norm[0,1]
        output_full_path_relev = output_path_relev / f'{clip_model.positives[k]}_{idx:0>5}'
        output = colormap_saving(valid_map[k].unsqueeze(-1), colormap_options,
                        output_full_path_relev) # H, W, 1

        # (lerf/langsplat-style) composited relvancy map
        # rel_map(>0.5) -> clip[0.5,1] -> norm[0,1] 
        p_i = torch.clip(valid_map[k] - 0.5, 0, 1).unsqueeze(-1)
        _, valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (valid_map[k] < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3
        output_full_path_compo = output_path_compo / f'{clip_model.positives[k]}_{idx:0>5}'
        colormap_saving(valid_composited, colormap_options, output_full_path_compo)
        
        # (lerf/langsplat-style) composited relvancy map with white background
        white_mask = torch.ones_like(image)
        output_full_path_lerf_compo = output_path_lerf_compo / f'{clip_model.positives[k]}_{idx:0>5}'
        show_result(valid_composited.cpu().numpy(), output_full_path_lerf_compo)
        
        # (GAGS-style) segment mask with composited image
        # rel_map -> norm[0,1] -> clip[0.5,1] -> norm[0,1] -> thresh(0.4) -> mask
        mask_pred = (output.cpu() > thresh).to(torch.uint8)
        mask_pred = smooth_GPU(mask_pred) # H, W
        mask_show = mask_pred.astype(bool)
        np_output = output.cpu().numpy() # H, W, 1
        avg_filtered = cv2.filter2D(np_output, -1, kernel) # H, W
        avg_filtered = torch.from_numpy(avg_filtered).unsqueeze(-1).to(valid_map.device) # H, W, 1
        _, valid_composited = colormaps.apply_colormap((0.5 * output + 0.5 * avg_filtered), colormaps.ColormapOptions("turbo"))
        
        valid_mask_composited=torch.zeros_like(image)
        valid_mask_composited[~mask_show, :] = image[~mask_show, :] * 0.4 + white_mask[~mask_show, :] * 0.1
        valid_mask_composited[mask_show, :] = valid_composited[mask_show, :] * 1.0
        output_full_path_mask_compo = output_path_mask_compo / f'{clip_model.positives[k]}_{idx:0>5}'
        colormap_saving(valid_mask_composited, colormap_options, output_full_path_mask_compo)

def lerf_compute_relvancy(dataset : ModelParams, 
                          iteration : int, 
                          pipeline : PipelineParams, 
                          cam_id : str, 
                          prompt : str, 
                          video_mode : bool, 
                          video_frames: int = 120):
    """
    Compute and visualize relevancy of text query with lerf-style and GAGS-style.
    Args:
        dataset (ModelParams): GS model parameters.
        iteration (int): Iteration number of GS scene.
        pipeline (PipelineParams): GS pipeline parameters.
        cam_id (str): View index for relevancy computation. Queries separated by ','. If None, test all cameras. 
        prompt (str): Text query for relevancy computation.
        video_mode (bool): If True, generate video frames using path interpolation.
        video_frames (int, optional): Number of interpolated views for video generation. Default 120 (5s*24fps).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1= "cpu"
    with torch.no_grad():       
        # path settings
        output_path = os.path.join(args.model_path, 'train', "ours_{}".format(args.iteration), "relvancy_heat_map")
        img_dir=os.path.join(args.source_path, 'images')
        img_paths=sorted(glob.glob(os.path.join(img_dir, '*')))
        os.makedirs(output_path, exist_ok=True)
        
        # load CLIP model
        clip_model = OpenCLIPNetwork_eval(device)
        prompt_list=prompt.split(',')
        clip_model.set_positives(prompt_list)
        
        # load GS scene
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False) # load scene info from .ply file
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)        
        camlist = scene.getTrainCameras()
        origin_feature_shape = (camlist[0].semantic_feature_height, camlist[0].semantic_feature_width)
        feature_out_dim=camlist[0].img_embed.shape[1] # 512
        
        # if no cam_id given, test for all cameras
        if cam_id == None:
            id_list=[i for i in range(len(camlist))]       
        else:            
            cam_id_list = cam_id.split(',')
            cam_id_list = [int(item) for item in cam_id_list]
            
            if video_mode:
                view_list = [camlist[id] for id in cam_id_list]
                
                inter_num = video_frames # 5s/24fps video
                RT_int = generate_interpolated_path(view_list, inter_num, n_interp_as_total=True, spline_degree=3)
                print("RT_interpolated.shape=",RT_int.shape)
                # sometimes the position of interpolated camera is wrong
                # thus use a simple scipy API result to replace it  
                T_new_list = simple_interpolation(view_list, inter_num, spline_degree=2)
                
                templete_cam = camlist[cam_id_list[0]]
                camlist_new=[]
                for i in range(RT_int.shape[0]):
                    R = RT_int[i, :3, :3]
                    R[:,1:3] = -R[:,1:3]
                    T = T_new_list[i]
                    # print(f"i:{i} --> R:{R} T:{T}")
                    cam_new = create_novel_view(R, T, templete_cam, render_h=origin_feature_shape[0], render_w=origin_feature_shape[1])
                    camlist_new.append(cam_new)
                id_list = [i + 10000 for i in range(len(camlist_new))] # just to distinguish from original cam_id
                camlist = camlist_new
            else:
                camlist = [camlist[id] for id in cam_id_list]
                id_list = cam_id_list
        
        colormap_options = colormaps.ColormapOptions(
            colormap="turbo",
            normalize=True,
            colormap_min=-1.0,
            colormap_max=1.0,
        )
        
        cnn_decoder_ckpt_path = os.path.join(dataset.model_path, "decoder_chkpnt{}.pth".format(iteration))
        
        if dataset.speedup:
            # feature_in_dim = int(feature_out_dim/32)
            feature_in_dim=16
            cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
            cnn_decoder_ckpt=torch.load(cnn_decoder_ckpt_path)
            if 'module_state_dict' in cnn_decoder_ckpt:
                cnn_decoder.load_state_dict(cnn_decoder_ckpt['module_state_dict'])
            else:
                cnn_decoder.load_state_dict(cnn_decoder_ckpt)        
        
        compressed_sem_feats = torch.zeros(len(id_list), *origin_feature_shape, feature_in_dim).to(device1)
        for j in range(len(id_list)):
            viewcam = camlist[j]
            viewcam.image_height=origin_feature_shape[0] # change to original feature map size
            viewcam.image_width=origin_feature_shape[1]
            render_pkg = render(viewcam, gaussians, pipeline, background)
            feature_map = render_pkg["render"] # 16,h,w
            if video_mode:
                render_rgb_pkg = render(viewcam, gaussians, pipeline, background, feature_mode = False)
                render_rgb = render_rgb_pkg["render"] # 3,h,w
                camlist[j].original_image = render_rgb
            # print('feature_map.shape=',feature_map.shape)
            compressed_sem_feats[j] = feature_map.permute(1,2,0).to(device1) # 1, num_eval_imgs, h, w, c=16
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()    
        
        for j, idx in enumerate(tqdm(id_list)):
            sem_feat = compressed_sem_feats[j, ...] # h, w, c=16
            sem_feat = sem_feat.float().to(device)
            if video_mode:
                rgb_img = camlist[j].original_image.permute(1,2,0).to(device)
            else:
                rgb_img = cv2.imread(img_paths[idx])[..., ::-1] # BGR->RGB h, w, c=3
                rgb_img = (rgb_img / 255.0).astype(np.float32)
                rgb_img = torch.from_numpy(rgb_img).to(device)

            restored_feat = cnn_decoder(sem_feat.permute(2,0,1))        # 512, h, w
            restored_feat = restored_feat.permute(1,2,0).unsqueeze(0)   # 1, h, w, 512
            
            activate_stream(restored_feat, rgb_img, clip_model, Path(output_path), idx, 
                                                thresh=0.4, colormap_options=colormap_options)
            del restored_feat
            torch.cuda.empty_cache() 
            
def pcd_relvancy(dataset : ModelParams, 
                 iteration : int, 
                 prompt : str, 
                 feature_dim : int, 
                 rel_thresh : float = 0.4, 
                 mask_color : Literal["rel", "default"] = "default", 
                 bg_color : Literal["RGB", "gray", "mix"] = "mix", 
                 save_pcd : bool = False):
    """
    Compute and visualize relevancy of text query with pointcloud.
    Args:
        dataset (ModelParams): GS model parameters.
        iteration (int): Iteration number of GS scene.
        prompt (str): Text query for relevancy computation.
        feature_dim (int): Dimension of the semantic feature.
        rel_thresh (float): Threshold for cosine relevancy computation.
        mask_color (str): Color for the selected(masked) point for visualization. 
        - rel: original relevancy visualizing color; default: red.
        bg_color (str): Color for the unselected point for visualization. 
        - RGB: original point color; gray: [0.5, 0.5, 0.5]; mix: mix of original point color and white.
        save_pcd (bool): Whether to save the pointcloud with visualized color.
    """
    
    with torch.no_grad():
        # get feature decoder
        cnn_decoder_ckpt_path = os.path.join(dataset.model_path, "decoder_chkpnt{}.pth".format(iteration))
        if dataset.speedup: 
            feature_in_dim = int(feature_dim/32)
            cnn_decoder = CNN_decoder(feature_in_dim, feature_dim)
            cnn_decoder.load_state_dict(torch.load(cnn_decoder_ckpt_path)['module_state_dict'])  
            
        # get CLIP model
        clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
        prompt_list=prompt.split(',')
        clip_model.set_positives(prompt_list) 
        
        # load pcd file
        pcd_path = os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
        plydata = PlyData.read(pcd_path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)       
        features_dc = np.zeros((xyz.shape[0], 3))
        features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
        rgb = features_dc*C0 + 0.5
        RGB_min = np.min(rgb, axis=0) 
        RGB_max = np.max(rgb, axis=0)
        
        # RGB normalization
        rgb = (rgb - RGB_min) / (RGB_max - RGB_min)
        
        semantic_feature=np.zeros((xyz.shape[0], 16))
        for i in range(16):
            semantic_feature[:, i] = np.asarray(plydata.elements[0]["semantic_" + str(i)])
        print("semantic_feature=",semantic_feature.shape)
        
        # 16->feature_dim
        semantic_feature = torch.from_numpy(semantic_feature).permute(1,0).unsqueeze(-1).to(torch.float32).cuda() # 16, N_points, 1
        if semantic_feature.shape[1] > 100_0000:
            pass
        else:
            semantic_feature_fulldim=cnn_decoder(semantic_feature) # feature_dim(512), N_points, 1     
            torch.cuda.empty_cache()
            semantic_feature_fulldim=semantic_feature_fulldim.squeeze(-1).permute(1,0) # N_points, feature_dim(512)
            
        # compute relevancy and visualize
        cmap = plt.get_cmap("viridis")
        norm = pltcolors.Normalize(vmin=0.2, vmax=0.7)
        for idx, prompt in enumerate(prompt_list):
            
            if semantic_feature.shape[1] > 100_0000:
                semantic_feature_group = torch.split(semantic_feature, 100_0000, dim=1)               
                relevancy_group=[]
                for i in range(len(semantic_feature_group)):
                    semantic_feature_fulldim = cnn_decoder(semantic_feature_group[i])
                    torch.cuda.empty_cache()
                    semantic_feature_fulldim=semantic_feature_fulldim.squeeze(-1).permute(1,0) # N_points(100_0000), feature_dim(512)
                    rel_part=clip_model.get_relevancy(semantic_feature_fulldim, idx)[:,0] # N_points, 1
                    relevancy_group.append(rel_part)
                relevancy=torch.cat(relevancy_group, dim=0)
            else:    
                relevancy=clip_model.get_relevancy(semantic_feature_fulldim, idx)[:,0] # N_points, 1
            
            # visualize normed relevancy
            colors = cmap(norm(relevancy.cpu().numpy().clip(0.2,0.7))) # N_points, 4
            rgb_colors_rel = colors[:, :3] # N_points, 3
            vis_pcd(xyz, rgb_colors_rel, window_name=str(prompt)) 
            
            relevancy = relevancy - torch.min(relevancy)
            relevancy = relevancy / (torch.max(relevancy) + 1e-9) 
            relevancy = relevancy * (1.0 - (-1.0)) + (-1.0)
            relevancy = torch.clip(relevancy, 0, 1)  
            mask = (relevancy > rel_thresh).cpu().numpy()
            
            # visualize binary smoothed GS kernel mask
            mask = smooth_pcd_mask(mask, xyz, radius=0.05, threshold=20)
            colors_seg = cmap(relevancy.cpu().numpy()) # N_points, 4
            rgb_colors_seg = np.zeros((xyz.shape[0], 3)) # N_points, 3
            if mask_color == "rel":
                rgb_colors_seg = colors_seg[:, :3] # original relevancy visualizing color
            elif mask_color == "default":
                rgb_colors_seg[:] = [1.0, 0.1, 0.05]  # red
            if bg_color == "RGB":
                rgb_colors_seg[~mask] = 1.0 * rgb[~mask] # original RGB color
            elif bg_color == "gray":
                rgb_colors_seg[~mask] = [0.5, 0.5, 0.5] # gray
            elif bg_color == "mix":
                white_mask = np.array([1.0, 1.0, 1.0]).reshape(1, 3).repeat(xyz.shape[0],axis=0)
                rgb_colors_seg[~mask] = 0.5 * rgb[~mask] + 0.3 * white_mask[~mask] 
            vis_pcd(xyz, rgb_colors_seg, window_name=str(prompt)+'_seg', point_size=3) 
            
            # save pointcloud with seg part color replaced
            if save_pcd:
                pcd_out_path = pcd_path.replace(".ply", "_{}.ply".format(prompt))
                for i in range(3):
                    rgb_colors_seg[:,i] = rgb_colors_seg[:,i] * (RGB_max[i] - RGB_min[i]) + RGB_min[i]
                plydata.elements[0]["f_dc_0"] = (rgb_colors_seg[:,0] - 0.5) / C0
                plydata.elements[0]["f_dc_1"] = (rgb_colors_seg[:,1] - 0.5) / C0
                plydata.elements[0]["f_dc_2"] = (rgb_colors_seg[:,2] - 0.5) / C0
                plydata.write(pcd_out_path)           
            
def compute_loss(dataset : ModelParams, 
                 iteration : int, 
                 pipeline : PipelineParams, 
                 cam_id:str):
    """
    Compute and visualize the loss between the GT feature map and the predicted feature map.
    Args:
        dataset (ModelParams): GS model parameters.
        iteration (int): Iteration number of GS scene.
        pipeline (PipelineParams): GS pipeline parameters.
        cam_id (str): View index for loss computation. Queries separated by ','. If None, test all cameras.
    """
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False) # load scene info from .ply file
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")   
        camlist=scene.getTrainCameras()
        cam_id_list=[i for i in range(len(camlist))]
        
        cnn_decoder_ckpt_path = os.path.join(dataset.model_path, "decoder_chkpnt{}.pth".format(iteration))    
        cnn_scale_decoder_ckpt_path = os.path.join(dataset.model_path, "scale_decoder_chkpnt{}.pth".format(iteration))
        if dataset.speedup:
            feature_out_dim=camlist[0].img_embed.shape[1] # 512
            # feature_in_dim = int(feature_out_dim/32)
            feature_in_dim = 16
            cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
            cnn_decoder.load_state_dict(torch.load(cnn_decoder_ckpt_path)['module_state_dict']) 
            
        cnn_scale_decoder=CNN_scale_decoder(feature_in_dim, 3)
        cnn_scale_decoder.load_state_dict(torch.load(cnn_scale_decoder_ckpt_path)['module_state_dict'])
        
        # if no cam_id given, test for all cameras
        if cam_id != None:
            cam_id_list = [int(item) for item in cam_id.split(',')]
            camlist=[camlist[id] for id in cam_id_list]  

        for idx, viewcam in tqdm(enumerate(camlist)):
            render_pkg = render(viewcam, gaussians, pipeline, background)
            feature_map = render_pkg["render"] 
            scale_map=cnn_scale_decoder(feature_map) 
            if dataset.speedup:
                feature_map = cnn_decoder(feature_map) 
            GT_feature_map, mask = read_sam_clip_feature(viewcam.img_embed.cuda(), viewcam.seg_map.cuda(),scale_map) # 512,H,W   
            GT_feature_map = GT_feature_map * mask
            feature_map = feature_map * mask
            # l2 loss along feature dimension
            feat_dim_l2_loss = torch.sqrt(torch.sum((GT_feature_map.cuda() - feature_map.cuda())**2, dim=0))  # H,W
            show_and_save_loss_map(feat_dim_l2_loss, 
                                    torch.mean(torch.abs(feature_map),dim=0), 
                                    torch.mean(torch.abs(GT_feature_map),dim=0), 
                                    Path(dataset.model_path), iteration, cam_id_list[idx])
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)  
    parser.add_argument("--quiet", action="store_true") 
    parser.add_argument("--cam_id", default=0)     
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--loss_mode", action="store_true") # compute loss between GT and predicted feature map
    parser.add_argument("--pcd_mode", action="store_true")  # compute text-feature relevancy in learned 3D space
    parser.add_argument("--image_mode", action="store_true")# compute text-feature relevancy in rendered 2D image
    parser.add_argument("--video", action="store_true")     # (for image mode) whether to interpolate density views 
                                                            # for video generating using given viewpoints
    args = get_combined_args(parser)
    
    safe_state(args.quiet)

    if args.pcd_mode==True:
        pcd_relvancy(model.extract(args), args.iteration, args.prompt, 512)
    elif args.loss_mode==True:
        compute_loss(model.extract(args), args.iteration, pipeline.extract(args), args.cam_id)
    elif args.image_mode==True: 
        lerf_compute_relvancy(model.extract(args), args.iteration, pipeline.extract(args), args.cam_id, args.prompt, args.video)    