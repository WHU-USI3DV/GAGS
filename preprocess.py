import os
import random
import argparse

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
###
from typing import Any, Dict, Generator, ItemsView, List, Tuple
import math
from itertools import product
from utils.SAM_utils import build_all_layer_mindepth_point_grids, build_mindepth_point_grid, sample_based_mapping

from torchvision import transforms
from torchvision.transforms import ToPILImage
from matplotlib import pyplot as plt
from PIL import Image
import copy
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from skimage import measure

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)
    
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

def build_all_layer_depth_point_grids( # 生成每level的层[归一化]格网点坐标
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

def project_from_sampled_pcd(pcd_pxl_mask, pcd_pxl_mapping, n_layers : int, height, width) -> List[np.ndarray]:
    """Project point to each imgs with mapping matrix."""
    points_by_layer = []
    for i in range(n_layers + 1):
        points=pcd_pxl_mapping[pcd_pxl_mask].astype(np.float32) # N_points,2
        points[:, 0]=points[:, 0]/height
        points[:, 1]=points[:, 1]/width
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

def create(image_list, data_list, save_folder, depth_mode, min_depth_mode, pcd_min_depth_mode, 
           depths_list=None, min_depth_list=None, pcd_mindepth_pth=None, mode_CLIP='default',model=None, preprocess=None, model_sam=None, sample_pts = None):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size=512
    seg_maps = []
    total_lengths = []
    timer = 0
    img_embeds = torch.zeros((len(image_list), 300, embed_size))
    seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:])) 
    
    if pcd_min_depth_mode:
        pcd_depth=np.load(os.path.join(pcd_mindepth_pth,'pcd_depth.npy')) # N_points
        pcd_pxl_mask=np.load(os.path.join(pcd_mindepth_pth,'pcd_pxl_mask.npy')) # N_points, N_cameras
        pcd_pxl_mapping=np.load(os.path.join(pcd_mindepth_pth,'pcd_pxl_mapping.npy')) # N_points, N_cameras, 2
        sample_num_pcd = round(0.02*pcd_depth.shape[0]) # select k% of original points
        sample_idx = sample_from_pcd(pcd_depth, pcd_pxl_mask, sample_num_pcd)
        print('pcd sample num:',len(sample_idx))
        h,w=image_list[0].shape[1:] 
        print('image shape:', h, w)
        
    elif not depth_mode : # default grid SAM
        mask_generator = SamAutomaticMaskGenerator(
            model=model_sam,
            points_per_side=32,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=1,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
        )
        mask_generator.predictor.model.to('cuda')
    else:
        pass

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        timer += 1
        if min_depth_mode:
            depth_sample_points,_ =build_all_layer_mindepth_point_grids(
                n_per_side=8,n_layers=0,scale_per_layer=1,nsample_min_distance=4,depth_map=depths_list[i],depth_sample=min_depth_list[i])
            print(f'sample points of img{i}:',depth_sample_points[0].shape[0])
            mask_generator = SamAutomaticMaskGenerator(
            model=model_sam,
            points_per_side=None,
            point_grids=depth_sample_points,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
            )
            mask_generator.predictor.model.to('cuda')
            
        elif depth_mode:
            depth_sample_points,_ =build_all_layer_depth_point_grids(
                n_per_side=8,n_layers=0,scale_per_layer=1,depth_map=depths_list[i])
            mask_generator = SamAutomaticMaskGenerator(
            model=model_sam,
            points_per_side=None,
            point_grids=depth_sample_points,
            pred_iou_thresh=0.7,
            box_nms_thresh=0.7,
            stability_score_thresh=0.85,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=100,
            )
            mask_generator.predictor.model.to('cuda')
        
        elif pcd_min_depth_mode:
            depth_sample_points = project_from_sampled_pcd(pcd_pxl_mask[sample_idx, i].astype(bool), pcd_pxl_mapping[sample_idx, i], n_layers=0, height=h, width=w)   
            print(f'sample points of img{i}:',depth_sample_points[0].shape[0])
            mask_generator = SamAutomaticMaskGenerator(
                model=model_sam,
                points_per_side=None, ### default
                point_grids=depth_sample_points, ### depth based
                pred_iou_thresh=0.7,
                box_nms_thresh=0.7,
                stability_score_thresh=0.85,
                crop_n_layers=0,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=100,
            )
            mask_generator.predictor.model.to('cuda')
            
        elif sample_pts is not None and len(sample_pts) > 0:
            depth_sample_points = [sample_pts[i]]
            mask_generator = SamAutomaticMaskGenerator(
                model=model_sam,
                points_per_side=None, ### default
                point_grids=depth_sample_points, ### depth based
                pred_iou_thresh=0.7,
                box_nms_thresh=0.7,
                stability_score_thresh=0.85,
                crop_n_layers=0,
                crop_n_points_downscale_factor=1,
                min_mask_region_area=100,
            )
            mask_generator.predictor.model.to('cuda')
                
        try:
            img_embed, seg_map = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder, mask_generator, model, preprocess, mode_CLIP)
        except:
            raise ValueError(timer)

        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        total_lengths.append(total_length)
        
        if total_length > img_embeds.shape[1]:
            pad = total_length - img_embeds.shape[1]
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(image_list), pad, embed_size))
            ], dim=1)

        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
        img_embeds[i, :total_length] = img_embed
        
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        seg_maps[i] = seg_map

    mask_generator.predictor.model.to('cpu')
        
    for i in range(img_embeds.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {
            'feature': img_embeds[i, :total_lengths[i]],
            'seg_maps': seg_maps[i]
        }
        sava_numpy(save_path, curr)

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())

def _embed_clip_sam_tiles(image, sam_encoder, mask_generator, model, preprocess, mode_CLIP):
    aug_imgs = torch.cat([image])
    # print('aug_imgs:',aug_imgs.shape,aug_imgs.device,aug_imgs.dtype)
    seg_images, seg_map = sam_encoder(aug_imgs, mask_generator, mode = mode_CLIP) # SAM segmentation
    
    clip_embeds = {}
    for mode in ['default', 's', 'm', 'l']:
        tiles = seg_images[mode] # default: touple(tensor(b,3,H,W)) 
        tiles = tiles.to("cuda")
        with torch.no_grad():
                
            if mode_CLIP == 'default':
                clip_embed = model.encode_image(tiles) # CLIP embedding
                clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
                clip_embeds[mode] = clip_embed.detach().cpu().half()
    
    return clip_embeds, seg_map

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

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

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
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr # mask包含的其他mask相对其自身不能太小
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        print("No masks with scores above threshold")
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        print("No masks with inner threshold")
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        print("No masks with inner threshold")
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

def sam_encoder(image, mask_generator, mode = 'default'):
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # print("cv2.imread shape",image.shape,type(image)) <class 'numpy.ndarray'>
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)

    # 每个 mask 输出是一个list，每个元素是一个dict，包含segmentation, area, predicted_iou, stability_score等信息
    # mask_update输出的是tuple，每个元素是一个dict
    
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
    
    if mode == 'default': # default
        # print("mask data",masks_default,type(masks_default)) # list [{'segmentation':,'area':,...},{dict},...]
        seg_images, seg_maps = {}, {}
        seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
        if len(masks_s) != 0:
            seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
        if len(masks_m) != 0:
            seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
        if len(masks_l) != 0:
            seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
        
        return seg_images, seg_maps

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    parser.add_argument('--depth_mode',action='store_true', default=False)
    parser.add_argument('--mindepth_mode',action='store_true', default=False)
    parser.add_argument('--pcd_mindepth_mode',action='store_true', default=False)
    parser.add_argument('--encoder_mode', type=str, default='default') 
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    depth_mode = args.depth_mode
    min_depth_mode = args.mindepth_mode
    pcd_min_depth_mode = args.pcd_mindepth_mode
    encoder_mode = args.encoder_mode
    if min_depth_mode:
        depth_mode=True
    img_folder = os.path.join(dataset_path, 'images')
    depth_folder= os.path.join(dataset_path, 'depths')
    depth_sample_folder= os.path.join(dataset_path, 'depths_sample')
    pcd_depth_sample_folder= os.path.join(dataset_path, 'pcd_depths_sample')
    data_list = os.listdir(img_folder)
    data_list.sort()

    preprocess = None
    if encoder_mode == 'default':
        model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
        sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
        
    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path) # H，W，C

        orig_w, orig_h = image.shape[1], image.shape[0]
        
        if args.resolution in [1, 2, 4, 8]:
            global_down=args.resolution
        
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
            
        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
        # print('data_path:',image_path)
        
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))] 
    imgs = torch.cat(images) # n, C, H, W  uint8
    # imgs = imgs[126:162] # for debug
    print('imgs:',imgs.shape) 
    
    depths = None
    min_depths = None
    if depth_mode:
        depth_list = []
        depth_data_list = os.listdir(depth_folder)
        depth_data_list.sort()
        for data_path in depth_data_list:
            depth_path = os.path.join(depth_folder, data_path)
            depth_image = torch.from_numpy(np.load(depth_path))
            depth_list.append(depth_image[None, ...])
            # print('depth_path:', depth_path,depth_image.shape)
        depths=torch.cat(depth_list) # n, H, W
        print('depths.shape=',depths.shape)
        
        if min_depth_mode:
            min_depth_list = []
            min_depth_data_list = os.listdir(depth_sample_folder)
            min_depth_data_list.sort()
            assert len(min_depth_data_list)==len(depth_list), "depth map number != min depth map number"
            for data_path in min_depth_data_list:
                depth_path = os.path.join(depth_sample_folder, data_path)
                min_depth_image = torch.from_numpy(np.load(depth_path))
                assert min_depth_image.shape[0]==depths.shape[1] and min_depth_image.shape[1]==depths.shape[2], "depth map shape != min depth map shape" 
                min_depth_list.append(min_depth_image[None, ...])
                # print('depth_path:', depth_path,depth_image.shape)
            min_depths=torch.cat(min_depth_list) # n, H, W
            print('min_depths.shape=',min_depths.shape)
        
    save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)
    create(imgs, data_list, save_folder, depth_mode, min_depth_mode, pcd_min_depth_mode,
           depths_list = depths, min_depth_list = min_depths, pcd_mindepth_pth = pcd_depth_sample_folder, 
           mode_CLIP = encoder_mode, model=model, preprocess=preprocess, model_sam=sam)