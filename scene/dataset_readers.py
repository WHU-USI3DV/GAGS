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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

    img_embed:torch.tensor = None
    seg_map:torch.tensor = None
    semantic_feature_height:int = None
    semantic_feature_width:int = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    semantic_feature_dim: int 

def read_sam_clip_feature(img_embed, seg_map, scale_map, max_mode:bool=False, median_mode:bool=False, show_scale_map:bool=False):
    _,h,w=seg_map.shape
    c=img_embed.shape[-1] # 512
    feature_map=torch.zeros(c,h,w)
    feature_map=feature_map.permute(1,2,0) # h,w,c
    _,h_scale,w_scale=scale_map.shape 

    x, y = torch.meshgrid(torch.arange(0, h), torch.arange(0, w),indexing='ij')
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    seg_s = seg_map[1][x, y].squeeze(-1).long() # h*w
    seg_m = seg_map[2][x, y].squeeze(-1).long()
    seg_l = seg_map[3][x, y].squeeze(-1).long()
    
    mask_s = (seg_s != -1).reshape(1, h, w) # 1,h,w
    mask_m = (seg_m != -1).reshape(1, h, w)
    mask_l = (seg_l != -1).reshape(1, h, w)
    mask = mask_s & mask_m & mask_l # 所有非-1的区域
    mask=F.interpolate(mask.float().unsqueeze(0), size=(h_scale, w_scale), mode='nearest').squeeze(0).to(torch.bool)
    feature_map_s = img_embed[seg_s].reshape(h, w, -1).permute(2, 0, 1) # c,h,w 
    feature_map_m = img_embed[seg_m].reshape(h, w, -1).permute(2, 0, 1) # c,h,w
    feature_map_l = img_embed[seg_l].reshape(h, w, -1).permute(2, 0, 1) # c,h,w
    # print('feature_map_s:',feature_map_s.shape,feature_map_s.dtype)
    feature_map_s =F.interpolate(feature_map_s.unsqueeze(0), size=(h_scale, w_scale), mode='bilinear', align_corners=True).squeeze(0)
    feature_map_m =F.interpolate(feature_map_m.unsqueeze(0), size=(h_scale, w_scale), mode='bilinear', align_corners=True).squeeze(0)
    feature_map_l =F.interpolate(feature_map_l.unsqueeze(0), size=(h_scale, w_scale), mode='bilinear', align_corners=True).squeeze(0)

    if max_mode==True:
        mask_s=F.interpolate(mask_s.float().unsqueeze(0), size=(h_scale, w_scale), mode='nearest').squeeze(0).to(torch.bool)
        mask_m=F.interpolate(mask_m.float().unsqueeze(0), size=(h_scale, w_scale), mode='nearest').squeeze(0).to(torch.bool)
        mask_l=F.interpolate(mask_l.float().unsqueeze(0), size=(h_scale, w_scale), mode='nearest').squeeze(0).to(torch.bool)
        max_idx=torch.argmax(scale_map, dim=0) # h,w
        scale_map=F.one_hot(max_idx, num_classes=3).permute(2,0,1).float() # 3,h,w
        feature_map = feature_map_s * scale_map[0] * mask_s + feature_map_m * scale_map[1] * mask_m + feature_map_l * scale_map[2] * mask_l
        mask=feature_map[0:1]!=0.0 # 1,h,w
        
    elif median_mode==True:
        seg_map=F.interpolate(seg_map.unsqueeze(0),size=(h_scale, w_scale),mode='nearest').squeeze(0)
        scale_bal_idx=1 # scale: s(1),m(2),l(3)
        mask_cur=seg_map[scale_bal_idx]!=-1
        min_idx=torch.min(seg_map[scale_bal_idx][mask_cur]).to(torch.int).item()
        max_idx=torch.max(seg_map[scale_bal_idx]).to(torch.int).item()

        scale_map_bal=scale_map.clone()
        for i in range(min_idx,max_idx+1): 
            idx_cur=seg_map[scale_bal_idx]==i 
            num_cur=torch.sum(idx_cur).item()
            idx_cur=idx_cur.unsqueeze(0).repeat(3,1,1) # 3,h,w
            pix_cur=scale_map[idx_cur].reshape(3,-1) # 3,n
            if pix_cur.numel() != 0:
                median_scale=torch.median(pix_cur,dim=1)[0]
                median_scale=median_scale/median_scale.sum()
                scale_map_bal[idx_cur] = median_scale.unsqueeze(-1).repeat(1,num_cur).reshape(-1) # 3,n -> 3*n
        
        if show_scale_map==True:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(scale_map.permute(1,2,0).detach().cpu().numpy())
            plt.subplot(1,2,2)
            plt.imshow(scale_map_bal.permute(1,2,0).detach().cpu().numpy())
            plt.show()
            plt.close()
            
        feature_map = feature_map_s * scale_map_bal[0] + feature_map_m * scale_map_bal[1] + feature_map_l * scale_map_bal[2]
        
    else:
        feature_map = feature_map_s * scale_map[0] + feature_map_m * scale_map[1] + feature_map_l * scale_map[2]
    return feature_map, mask 

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, semantic_feature_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path) #(1908, 1423)
        
        if semantic_feature_folder == None:
            # print('No samentic feature found!')
            pass
        elif os.path.exists(os.path.join(semantic_feature_folder, image_name)+'_f.npy'):
            img_embed_path=os.path.join(semantic_feature_folder, image_name)+'_f.npy'
            segment_map_path=os.path.join(semantic_feature_folder, image_name)+'_s.npy'
            img_embed=torch.from_numpy(np.load(img_embed_path)) # (num_img_embeds, 512)
            seg_map=torch.from_numpy(np.load(segment_map_path)) # (4, h, w)
        else:
            assert False, "Semantic feature file not found!"

        if semantic_feature_folder == None:
            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height,
                                ) 
        else:
            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height,
                                img_embed=img_embed,
                                seg_map=seg_map,
                                semantic_feature_height=seg_map.shape[1],
                                semantic_feature_width=seg_map.shape[2]) 
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, foundation_model, images, eval, llffhold=8): # 读取scene信息
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    reading_dir = "images" if images == None else images

    if foundation_model =='sam_clip': # switch the feature type
        semantic_feature_dir='language_features'
        semantic_feature_folder = os.path.join(path, semantic_feature_dir)
        print("using foundation model:", foundation_model, "features from:", semantic_feature_folder)
    elif foundation_model =='none':
        semantic_feature_folder = None
    
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir), semantic_feature_folder=semantic_feature_folder) #读取每张图的相机参数和语义特征
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name) # camera info按照文件名排序

    if foundation_model != "none":
        semantic_feature_dim = cam_infos[0].img_embed.shape[1] # default cam_infos[0].semantic_feature.shape[0]
    else:
        semantic_feature_dim = 0
        
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 2] # avoid 1st to be test view
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 2] 
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos, # 包含语义特征
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim) # 特征维度
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, semantic_feature_folder, extension=".png"): 
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            
            semantic_feature_path = os.path.join(semantic_feature_folder, image_name) + '_fmap_CxHxW.pt' 
            semantic_feature_name = os.path.basename(semantic_feature_path).split(".")[0]
            semantic_feature = torch.load(semantic_feature_path)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                              semantic_feature=semantic_feature,
                              semantic_feature_path=semantic_feature_path,
                              semantic_feature_name=semantic_feature_name)) 
            
    return cam_infos

def readNerfSyntheticInfo(path, foundation_model, white_background, eval, extension=".png"): 
    if foundation_model =='sam':
        semantic_feature_dir = "sam_embeddings" 
    elif foundation_model =='lseg':
        semantic_feature_dir = "rgb_feature_langseg" 

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, semantic_feature_folder=os.path.join(path, semantic_feature_dir)) 
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, semantic_feature_folder=os.path.join(path, semantic_feature_dir)) 
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    semantic_feature_dim = train_cam_infos[0].semantic_feature.shape[0] 
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim) 
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
