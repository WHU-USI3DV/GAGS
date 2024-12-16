'''
Get the minimum depth of each image for SAM sampling

input: Gaussian point cloud, colmap camera intrinsics and extrinsics, Gaussian field rendered depth map
output: density field for SAM sampling
'''

import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
import math
import warnings
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as pltcolors


class PointCloudToImageMapper(object):
    def __init__(self, image_dim,
            visibility_threshold=0.25, cut_bound=0, intrinsics=None):
        
        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

    def compute_mapping(self, world_to_camera, coords, depth=None, intrinsic=None):
        """
        :param world_to_camera: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        
        device = 'cpu'
        world_to_camera=world_to_camera.to(device)
        coords = coords.to(device)
        depth = depth.to(device)
        
        if self.intrinsics is not None: # global intrinsics
            intrinsic = self.intrinsics.to(device)

        mapping = torch.zeros((3, coords.shape[0]), dtype=torch.int32)
        coords_new = torch.cat([coords, torch.ones([coords.shape[0], 1]).to(device)], dim=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        p = torch.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2] # u(W)
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2] # v(H)
        pi = torch.round(p).to(torch.int32) # simply round the projected coordinates
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                    * (pi[0] < self.image_dim[0]-self.cut_bound) \
                    * (pi[1] < self.image_dim[1]-self.cut_bound)

        if depth is not None:
            depth_mapdata = p[2][inside_mask] # N_points_inside
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]] # N_points_inside
            diff_depth = torch.abs(depth_cur - depth_mapdata)
            occlusion_mask =  diff_depth <= self.vis_thres * depth_cur
            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2]>0 # make sure the depth is in front
            inside_mask = front_mask*inside_mask

        mapping[0][inside_mask] = pi[1][inside_mask] # v
        mapping[1][inside_mask] = pi[0][inside_mask] # u
        mapping[2][inside_mask] = 1

        return mapping.T
    
def vis_one_image_pcd(gs_pcd, w2c_RT, K, depth_map):
    '''
    gs_pcd: [N_points, 3]
    w2c_RT: [4, 4]
    K: [3, 3]
    depth_map: [H, W]
    '''
    # if the device is not cuda, print warning
    device="cuda"
    
    if gs_pcd.device.type != device or w2c_RT.device.type != device \
    or K.device.type != device or depth_map.device.type != device:
        warnings.warn(f"Devices:{gs_pcd.device},{w2c_RT.device},{K.device},{depth_map.device} are not all cuda, which may cause issues.", UserWarning)
    
    # generate the meshgrid of (x,y) for the depth map
    H,W = depth_map.shape[:2]
    u_map = torch.arange(W, device=device).repeat(H, 1) # [H, W]
    v_map = torch.arange(H, device=device).repeat(W, 1).T # [H, W]
    axis_map = torch.stack([v_map,u_map],dim=2) # [H, W, 2]
    axis_map = torch.cat([axis_map,torch.ones([H,W,1],device=device)],dim=2) # [H, W, 3]
    coords_x_cam = (u_map - K[0, 2]) * depth_map / K[0, 0] # [H, W]
    coords_y_cam = (v_map - K[1, 2]) * depth_map / K[1, 1]
    coords_cam = torch.stack([coords_x_cam, coords_y_cam, depth_map], dim=0).reshape(3, -1) # [3, H*W]
    coords_cam = torch.cat([coords_cam,torch.ones([1,H*W],device=device)],dim=0) # [4, H*W]
    c2w_RT=torch.inverse(w2c_RT)
    coords_world=torch.matmul(c2w_RT,coords_cam)[:3] # [3, H*W]
    coords_world=coords_world.permute(1, 0) # [H*W, 3]
    # Another way to calculate c-to-w
    # c2w_R=torch.inverse(w2c_RT[:3,:3])
    # coords_world=torch.matmul(c2w_R, coords_cam[:3] - w2c_RT[:3,3].unsqueeze(-1))  # [3, H*W]
    # coords_world=coords_world.permute(1, 0) # [H*W, 3]
    print('coords_world:',coords_world.shape)
    vis_pcd_depth(coords_world.cpu().numpy(), depth_map.reshape(-1).cpu().numpy(), xyz_whole=gs_pcd.cpu().numpy())
    
def save_pcd_depth(pcd_depth, pcd_pxl_mask, pcd_pxl_mapping, cam_list, save_path, save_path_pcd=None):
    cam_H=cam_list[0].image_height
    cam_W=cam_list[0].image_width
    pcd_pxl_mask = pcd_pxl_mask.to(torch.bool)
    print('pcd_pxl_mask',pcd_pxl_mask.shape) # N_points, N_cameras
    print('pcd_pxl_mapping:',pcd_pxl_mapping.shape) # N_points, N_cameras, 2
    for cid in tqdm(range(len(cam_list))):
        cam_name=cam_list[cid].image_name
        depth_sample=torch.zeros((cam_H,cam_W))
        inview_mask_cur=pcd_pxl_mask[:,cid]
        depth_sample[pcd_pxl_mapping[inview_mask_cur,cid,0],pcd_pxl_mapping[inview_mask_cur,cid,1]] = pcd_depth[inview_mask_cur]
        
        # SAVE 
        save_path_cur=os.path.join(save_path, cam_name + '_depth_sample.npy')  
        np.save(save_path_cur,depth_sample.numpy())
    print(f'successfully saved depth samples ({len(cam_list)} * {depth_sample.shape}) to {save_path} !')
    
    # np.save(os.path.join(save_path_pcd,'pcd_depth.npy'),pcd_depth.numpy())
    # np.save(os.path.join(save_path_pcd,'pcd_pxl_mask.npy'),pcd_pxl_mask.numpy())
    # np.save(os.path.join(save_path_pcd,'pcd_pxl_mapping.npy'),pcd_pxl_mapping.numpy())
    # print(f'successfully saved pcd_depth ({pcd_depth.shape}) to {save_path_pcd}/pcd_depth.npy !')
    # print(f'successfully saved pcd_pxl_mask ({pcd_pxl_mask.shape}) to {save_path_pcd}/pcd_pxl_mask.npy !')
    # print(f'successfully saved pcd_pxl_mapping ({pcd_pxl_mapping.shape}) to {save_path_pcd}/pcd_pxl_mapping.npy !') 
    
def vis_pcd_depth(xyz,depth,window_name="Feature Point Cloud",xyz_whole=None):
    
    vmin=0
    vmax=0.3
    norm = pltcolors.Normalize()
    norm_depth = norm(depth)
    norm_depth = (norm_depth - vmin) / (vmax - vmin)
    norm_depth = np.clip(norm_depth, 0, 1)
    
    print('depth:',norm_depth.shape)
    cmap_obj=cm.get_cmap('rainbow')
    cmap_obj_reversed = cmap_obj.reversed()
    rgb = cmap_obj_reversed(norm_depth)[:,:3]
    print('xyz',xyz.shape,'rgb:',rgb.shape)
    
    # visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.get_render_option().point_size = 4
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    pcd=o3d.open3d.geometry.PointCloud()
    pcd.points= o3d.open3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    vis.add_geometry(pcd)

    if xyz_whole is not None:
        pcd_whole=o3d.open3d.geometry.PointCloud()
        pcd_whole.points= o3d.open3d.utility.Vector3dVector(xyz_whole)
        pcd_whole.colors = o3d.utility.Vector3dVector(np.ones_like(xyz_whole))
        vis.add_geometry(pcd_whole)
        
    vis.run()
    vis.destroy_window()
    
def mapping_img_2_pcd(pose, intrinsic, pcd, depth):
    H,W = depth.shape[:2]
    point2img_mapper = PointCloudToImageMapper(
        image_dim=(W,H), intrinsics=intrinsic)
    mapping=point2img_mapper.compute_mapping(world_to_camera=pose, coords=pcd, depth=depth)
    # print("mapping:",mapping.shape)
    
    return mapping
    
def process_one_image(gs_pcd, w2c_RT, K, depth_map):
    '''
    gs_pcd: [N_points, 3]
    w2c_RT: [4, 4]
    K: [3, 3]
    depth_map: [H, W]
    '''
    # if the device is not cuda, print warning
    device="cuda"
    
    if gs_pcd.device.type != device or w2c_RT.device.type != device \
    or K.device.type != device or depth_map.device.type != device:
        warnings.warn(f"Devices:{gs_pcd.device},{w2c_RT.device},{K.device},{depth_map.device} are not all cuda, which may cause issues.", UserWarning)
    
    # calculate the 3d-2d mapping based on the depth 
    n_points=gs_pcd.shape[0]
    mapping = torch.ones([n_points, 4], dtype=torch.int32) # 4：[1, v, u, mask]
    mapping[:, 1:4] = mapping_img_2_pcd(w2c_RT, K, gs_pcd, depth_map) # cpu
    mapping = mapping.to(device)
    mask = mapping[:, 3] 
    pcd_3d_mapping = mapping[:, 1:3] # [N, 2] 2->(v,u)
    depth_2d_3d = depth_map[mapping[:, 1], mapping[:, 2]] # [N] 
    depth_2d_3d[mask==0] = torch.inf
    return depth_2d_3d.cpu().numpy(), mask.cpu().numpy(), pcd_3d_mapping.cpu().numpy()
    
def main(model_params: ModelParams, iteration: int, sample_rate: float):
    
    depth_path = os.path.join(model_params.model_path, 'train', "ours_{}".format(iteration), "depths")
    depth_list = os.listdir(depth_path)
    depth_list.sort()
    depth_fullname_list=[os.path.join(depth_path, depth_list[i]) for i in range(len(depth_list))]
    
    # save_path=os.path.join(model_params.model_path, 'train', "ours_{}".format(iteration), "depths_sample")
    save_path = os.path.join(model_params.source_path, 'depths_sample')
    os.makedirs(save_path,exist_ok=True)
    # save_path_pcd=os.path.join(model_params.model_path, 'train', "ours_{}".format(iteration), "pcd_depths_sample")
    save_path_pcd = os.path.join(model_params.source_path, 'pcd_depths_sample')
    os.makedirs(save_path_pcd,exist_ok=True)    
    
    device="cuda"
    
    with torch.no_grad():
        gaussians = GaussianModel(model_params.sh_degree)
        scene = Scene(model_params, gaussians, load_iteration=iteration, shuffle=False) 
        sample_camera = scene.getTrainCameras()
        gs_pcd = gaussians.get_xyz
        # print(gs_pcd.shape)
        
        pcd_pxl_depth_list=[]
        pcd_pxl_mask_list=[]
        pcd_pxl_mapping_list=[]
        for cid in tqdm(range(len(sample_camera))):
            viewpoint_camera = sample_camera[cid]
            # print('cid:',cid, 'viewpoint_camera:',viewpoint_camera.image_name)
            w2c_RT = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
            tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
            focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
            focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
            K = torch.tensor(
                [
                    [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                    [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                    [0, 0, 1],
                ],
                device = device,
            )
            depth_map=torch.from_numpy(np.load(depth_fullname_list[cid])).to(device)
            pcd_pel_depth_cur, mask_cur, mapping_cur = process_one_image(gs_pcd, w2c_RT, K, depth_map)
            pcd_pxl_depth_list.append(pcd_pel_depth_cur)
            pcd_pxl_mask_list.append(mask_cur)
            pcd_pxl_mapping_list.append(mapping_cur)   
            # visualization the pointcloud and depth of one image 
            # vis_one_image_pcd(gs_pcd, w2c_RT, K, depth_map)
            
        pcd_pxl_depth = torch.from_numpy(np.stack(pcd_pxl_depth_list, axis=0)).permute(1,0) # [N_points, N_cameras]
        pcd_pxl_mask = torch.from_numpy(np.stack(pcd_pxl_mask_list, axis=0)).permute(1,0) # [N_points, N_cameras]
        pcd_pxl_mapping=torch.from_numpy(np.stack(pcd_pxl_mapping_list, axis=0)).permute(1,0,2) # [N_points, N_cameras, 2]
        print("pcd_pxl_depth:",pcd_pxl_depth.shape)
        print("pcd_pxl_mask:",pcd_pxl_mask.shape)
        
        point_ids = torch.unique(pcd_pxl_mask.nonzero(as_tuple=False)[:, 0]) # [N_points] 有对应2D pixel的3D points index
        pcd_min_depth = torch.min(pcd_pxl_depth, dim=1)[0] # [N_points]

        print('max_depth:',torch.max(pcd_min_depth[point_ids]),'min_depth:',torch.min(pcd_min_depth[point_ids]))
        
        # visualization
        # vis_pcd_depth(gs_pcd[point_ids].cpu().numpy(),pcd_min_depth[point_ids])
        
        # saving
        save_pcd_depth(pcd_min_depth, pcd_pxl_mask, pcd_pxl_mapping, sample_camera, save_path, save_path_pcd=save_path_pcd)
        
if __name__ == "__main__":
    parser = ArgumentParser(description="Depth based SAM sampling script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument('--sample_rate','-sr', type=float, default = 1.0, 
                        help='The number of sampling points in space per unit distance from the camera')
    args = get_combined_args(parser)
    print('args:',args.__dict__)
    
    iteration = args.iterations
    main(model.extract(args), iteration, args.sample_rate)