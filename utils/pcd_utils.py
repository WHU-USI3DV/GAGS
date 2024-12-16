import open3d as o3d
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import matplotlib.cm as cm
import matplotlib.colors as pltcolors
import matplotlib.pyplot as plt
from scene.cameras import Camera
import warnings
import math
from scipy.spatial import KDTree
from tqdm import tqdm

def vis_pcd(xyz,rgb,window_name="Feature Point Cloud",xyz_whole=None,point_size=2):
    
    norm = pltcolors.Normalize()
    norm_depth = norm(rgb)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.get_render_option().point_size = point_size
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    if xyz_whole is not None:
        pcd_whole=o3d.open3d.geometry.PointCloud()
        pcd_whole.points= o3d.open3d.utility.Vector3dVector(xyz_whole)
        pcd_whole.colors = o3d.utility.Vector3dVector(np.ones_like(xyz_whole)*0.2)
        vis.add_geometry(pcd_whole)
    
    # create point cloud object
    pcd=o3d.open3d.geometry.PointCloud()
    pcd.points= o3d.open3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    vis.add_geometry(pcd)
        
    vis.run()
    vis.destroy_window()
    
def vis_pcd_label(xyz,labels,window_name="Feature Point Cloud",xyz_whole=None,normal_xyz=None,normals=None,rect=None,cam_pos=None,
                  w2c_R_offset=None, w2c_T_offset=None):
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.get_render_option().point_size = 4
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    
    if xyz_whole is not None:
        pcd_whole=o3d.open3d.geometry.PointCloud()
        pcd_whole.points= o3d.open3d.utility.Vector3dVector(xyz_whole)
        pcd_whole.colors = o3d.utility.Vector3dVector(np.ones_like(xyz_whole)*0.2)
        vis.add_geometry(pcd_whole)
    
    if normal_xyz is not None and normals is not None:
        line_set = o3d.geometry.LineSet()
        points = np.concatenate([normal_xyz,normal_xyz+normals*3],axis=0)
        line_set.points = o3d.utility.Vector3dVector(points)
        lines = np.stack([np.arange(normal_xyz.shape[0]),np.arange(normal_xyz.shape[0])+normal_xyz.shape[0]],axis=0).T # [N,2]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(np.array([[1,0,0]]).repeat(normal_xyz.shape[0],axis=0)) # N,3
        vis.add_geometry(line_set)
    
    if rect is not None:
        cluster_num,rect_num = rect.shape[0],rect.shape[1]
        line_set_2 = o3d.geometry.LineSet()
        points_2_list=[]
        
        for i in range(cluster_num):
            points_2 = np.concatenate([rect[i],rect[i][0:1]],axis=0)
            points_2_list.append(points_2)
        points_2_list=np.concatenate(points_2_list,axis=0)
        line_set_2.points = o3d.utility.Vector3dVector(points_2_list)

        lines_list=[]
        for i in range(cluster_num):
            lines_2 = np.stack([np.arange(rect_num)+i*(rect_num+1),np.arange(rect.shape[1])+1+i*(rect_num+1)],axis=0).T
            lines_list.append(lines_2)
        lines_list=np.concatenate(lines_list,axis=0)
        line_set_2.lines = o3d.utility.Vector2iVector(lines_list)

        line_set_2.colors = o3d.utility.Vector3dVector(np.array([[0,1,0]]).repeat(lines_list.shape[0],axis=0)) # N,3
        vis.add_geometry(line_set_2)
        
    if cam_pos is not None:

        line_set_3 = o3d.geometry.LineSet()

        points_3_list=[]
        for i in range(cluster_num):
            
            point_3=np.concatenate([cam_pos[i:i+1],rect[i]],axis=0) # [rect_num+1,3]
            points_3_list.append(point_3)
        points_3_list=np.concatenate(points_3_list,axis=0)
        line_set_3.points = o3d.utility.Vector3dVector(points_3_list)

        line_list=[]
        for i in range(cluster_num):
            lines_3=np.stack([np.zeros((rect_num))+i*(rect_num+1),np.arange(rect_num)+1+i*(rect_num+1)],axis=1)
            line_list.append(lines_3)
        line_list=np.concatenate(line_list,axis=0)
        line_set_3.lines = o3d.utility.Vector2iVector(line_list)

        line_set_3.colors = o3d.utility.Vector3dVector(np.array([[0,1,0]]).repeat(line_list.shape[0],axis=0)) # N,3
        vis.add_geometry(line_set_3)
    
    if w2c_R_offset is not None and w2c_T_offset is not None:

        line_set_4 = o3d.geometry.LineSet()

        points_4_list=[]
        for i in range(cluster_num):
            camcenter_offset_c=np.zeros(3).astype(np.float32)
            cam_axis_offset_c=np.array([0,0,1]).astype(np.float32)
            camcenter_offset_w=np.matmul(w2c_R_offset[i].T, camcenter_offset_c - w2c_T_offset[i])
            cam_axis_offset_w=np.matmul(w2c_R_offset[i].T, cam_axis_offset_c - w2c_T_offset[i])
            # print('camcenter_offset_w:',camcenter_offset_w,'cam_axis_offset_w:',cam_axis_offset_w)
            point_4=np.stack([camcenter_offset_w,cam_axis_offset_w],axis=0) # [2,3]
            points_4_list.append(point_4)
        points_4_list=np.concatenate(points_4_list,axis=0)
        line_set_4.points = o3d.utility.Vector3dVector(points_4_list)

        lines_4=np.arange(cluster_num*2).reshape(-1,2)
        line_set_4.lines = o3d.utility.Vector2iVector(lines_4)

        line_set_4.colors = o3d.utility.Vector3dVector(np.array([[0,0,1]]).repeat(lines_4.shape[0],axis=0)) # N,3
        vis.add_geometry(line_set_4)    
            
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1)) # [N,4]
    colors[labels < 0] = 1 # set invalid points to black

    # create point cloud object
    pcd=o3d.open3d.geometry.PointCloud()
    pcd.points= o3d.open3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    vis.add_geometry(pcd)
        
    vis.run()
    vis.destroy_window()  
    
def compute_normal(xyz):
    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points = o3d.open3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=100))
    normals = np.asarray(pcd.normals)
    return normals

def pcd_Euclidean_Clustering(xyz,eps=0.2,min_points=30):
    pcd = o3d.open3d.geometry.PointCloud()
    pcd.points = o3d.open3d.utility.Vector3dVector(xyz)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)) # [N]
    max_label = labels.max()
    # print(f"point cloud{xyz.shape} has {max_label+1} clusters")
    return labels, max_label+1

def rotation_matrix_from_vectors(vec1, vec2):

    a = vec1 / torch.norm(vec1)
    b = vec2 / torch.norm(vec2)
    
    # Calculate the rotation axis (via cross product)
    v = torch.cross(a, b)
    
    # Calculate the angle between vectors
    c = torch.dot(a, b)
    s = torch.norm(v)
    
    # If the angle is 0 or 180 degrees, the rotation matrix is the identity matrix
    if torch.isclose(s, torch.tensor(0.0)):
        return torch.eye(3)
    
    # Construct the rotation matrix
    vx = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = torch.eye(3) + vx + torch.matmul(vx, vx) * ((1 - c) / (s ** 2))
    
    return rotation_matrix

def create_novel_view(w2c_R,w2c_T,old_view, render_h=None, render_w=None, image=None, img_embed=None, seg_map=None):
    old_uid=old_view.uid
    old_colmap_id=old_view.colmap_id
    old_FoVx,old_FoVy=old_view.FoVx,old_view.FoVy
    old_img_name=old_view.image_name
    old_semantic_feature_height,old_semantic_feature_width=old_view.semantic_feature_height,old_view.semantic_feature_width
    old_data_device=old_view.data_device
    if render_h is None or render_w is None:
        render_h,render_w=old_view.original_image.shape[1],old_view.original_image.shape[2]
    if image is None:
        image=torch.zeros((3,render_h,render_w),dtype=torch.float32)
    else:
        image=torch.nn.functional.interpolate(image.unsqueeze(0), size=(render_h, render_w), mode='bilinear').squeeze(0)
    if seg_map is not None:
        seg_map=torch.nn.functional.interpolate(seg_map.unsqueeze(0), size=(render_h, render_w), mode='nearest').squeeze(0)
        
    return Camera(colmap_id=old_colmap_id, R=w2c_R, T=w2c_T, 
                  FoVx=old_FoVx, FoVy=old_FoVy, 
                  image=image, gt_alpha_mask=None,
                  image_name=old_img_name, uid=old_uid, 
                  semantic_feature_height=old_semantic_feature_height, 
                  semantic_feature_width=old_semantic_feature_width, 
                  img_embed=img_embed, seg_map=seg_map,
                  data_device=old_data_device)

def smooth_pcd_mask(mask, xyz, radius=0.1, threshold=10):
    # mask: [N]
    # xyz: [N,3]
    print("start building KDTree")
    tree = KDTree(xyz)
    smoothed_mask = mask.copy()
    print("start smoothing mask")
    for i in tqdm(range(len(xyz))):
        indices = tree.query_ball_point(xyz[i], r=radius)
        valid_neighbors = np.sum(mask[indices])
        if valid_neighbors > threshold:
            smoothed_mask[i] = True
        elif valid_neighbors < 10:
            smoothed_mask[i] = False
    
    return smoothed_mask

def pcd_2_map(w2c_RT, K, depth_map, min_pos, max_pos):
    '''
    gs_pcd: [N_points, 3]
    w2c_RT: [4, 4]
    K: [3, 3]
    depth_map: [H, W]
    '''
    # if the device is not cuda, print warning
    device="cuda"
    
    if w2c_RT.device.type != device or K.device.type != device or depth_map.device.type != device:
        warnings.warn(f"Devices:{w2c_RT.device},{K.device},{depth_map.device} are not all cuda, which may cause issues.", UserWarning)
    
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
    # normalize the coordinates to [-1,1]
    coords_world_norm = -1 + 2 * (coords_world-min_pos)/(max_pos-min_pos) # [3, H*W] of [-1,1]
    coords_world_norm = coords_world_norm.reshape(3, H, W) # [3, H, W]
    
    return coords_world_norm, coords_world

def position_encoding(position_map, level=4):
    '''
    position_map: [3, H, W]
    '''
    PI = math.pi
    position_encoded_list=[]
    for l in range(level):
        position_encoded = torch.cat([torch.sin(PI * position_map * (2.0 ** l)), torch.cos(PI * position_map * (2.0 ** l))], dim=0) # [6, H, W]
        position_encoded_list.append(position_encoded)
    position_encoded = torch.cat(position_encoded_list, dim=0) # [6*level, H, W]

    return position_encoded