#!/usr/bin/env python
from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
from eval import colormaps
from models.networks import CNN_decoder, CNN_scale_decoder
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from gaussian_renderer import render
from eval.openclip_encoder import OpenCLIPNetwork
from eval.utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
import matplotlib
matplotlib.use('Agg')   # opencv-python has a crash with matplotlib(interactive backend),
                        # uninstall PyQt5 may solve this problem

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)
        
        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1 # name-1
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)           # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask
            
            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths


def activate_stream(sem_map, 
                    image, 
                    clip_model, 
                    image_name: Path = None,
                    img_ann: Dict = None, 
                    thresh : float = 0.5, 
                    colormap_options = None):
    valid_map = clip_model.get_max_across(sem_map)                 # 1xkxHxW
    valid_map = valid_map.squeeze(0)                               # kxHxW
    n_prompt, h, w = valid_map.shape

    # positive prompts
    iou_list = []
    for k in range(n_prompt):

        mask_lvl = np.zeros((h, w))
        # heatmap visualization
        scale = 30
        kernel = np.ones((scale,scale)) / (scale**2)
        np_relev = valid_map[k].cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev, -1, kernel) 
        avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
        valid_map[k] = 0.5 * (avg_filtered + valid_map[k])
        output_path_relev = image_name / 'heatmap' / f'{clip_model.positives[k]}'
        output_path_relev.parent.mkdir(exist_ok=True, parents=True)
        colormap_saving(valid_map[k].unsqueeze(-1), colormap_options,
                        output_path_relev)
    
        # lerf-style composited heatmap
        # p_i = torch.clip(valid_map[k] - 0.5, 0, 1).unsqueeze(-1)
        # valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        # mask = (valid_map[k] < 0.5).squeeze()
        # valid_composited[mask, :] = image[mask, :] * 0.3
        # output_path_compo = image_name / 'composited' / f'{clip_model.positives[k]}'
        # output_path_compo.parent.mkdir(exist_ok=True, parents=True)
        # colormap_saving(valid_composited, colormap_options, output_path_compo)
        
        # lerf-style composited heatmap with white mask
        white_mask = torch.ones_like(image)
        # valid_lerf_composited = torch.zeros_like(image)
        # valid_lerf_composited[mask, :] = image[mask, :] * 0.3 + white_mask[mask, :] * 0.3
        # valid_lerf_composited[~mask, :] = valid_composited[~mask, :] * 0.7 + white_mask[~mask, :] * 0.3
        # output_path_lerf_compo = image_name / 'lerf_composited' / f'{clip_model.positives[k]}'
        # output_path_lerf_compo.parent.mkdir(exist_ok=True, parents=True)
        # show_result(valid_lerf_composited.cpu().numpy(), output_path_lerf_compo)
        
        # truncate the heatmap into mask 
        output = valid_map[k]
        output = output - torch.min(output)
        output = output / (torch.max(output) + 1e-9)
        output = output * (1.0 - (-1.0)) + (-1.0)
        output = torch.clip(output, 0, 1)

        mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
        mask_pred = smooth(mask_pred)
        mask_lvl = mask_pred 
        mask_gt = img_ann[clip_model.positives[k]]['mask'].astype(np.uint8)
        
        # mask visualization 
        mask_show = mask_pred.astype(bool)
        np_output = output.unsqueeze(0).cpu().numpy() # 1,H,W
        avg_filtered = cv2.filter2D(np_output.transpose(1,2,0), -1, kernel) # H, W
        avg_filtered = torch.from_numpy(avg_filtered).unsqueeze(-1).to(valid_map.device) # H, W, 1
        _, valid_composited = colormaps.apply_colormap((0.5 * output.unsqueeze(-1) + 0.5 * avg_filtered), colormaps.ColormapOptions("turbo"))
        
        valid_mask_composited=torch.zeros_like(image)
        valid_mask_composited[~mask_show, :] = image[~mask_show, :] * 0.4 + white_mask[~mask_show, :] * 0.1
        valid_mask_composited[mask_show, :] = valid_composited[mask_show, :] * 1.0 + white_mask[mask_show, :] * 0.0
        output_path_mask_compo = image_name / 'mask_composited' / f'{clip_model.positives[k]}'
        output_path_mask_compo.parent.mkdir(exist_ok=True, parents=True)
        show_result(valid_mask_composited.cpu().numpy(), output_path_mask_compo)
        
        # calculate iou 
        intersection = np.sum(np.logical_and(mask_gt, mask_pred))
        union = np.sum(np.logical_or(mask_gt, mask_pred))
        iou = np.sum(intersection) / np.sum(union)
        
        iou_list.append(iou)
        
        # save for visulsization
        save_path = image_name / f'chosen_{clip_model.positives[k]}.png'
        vis_mask_save(mask_lvl, save_path)

    return iou_list


def lerf_localization(sem_map, image, clip_model, image_name, img_ann):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)

    valid_map = clip_model.get_max_across(sem_map)   # 1,H,W,512 -> 1, n_phrases, H, W
    
    # positive prompts
    acc_num = 0
    positives = list(img_ann.keys())
    for k in range(len(positives)):
        select_output = valid_map[:, k] # 1, H, W

        scale = 30
        kernel = np.ones((scale,scale)) / (scale**2)
        np_relev = select_output.cpu().numpy()

        avg_filtered = cv2.filter2D(np_relev.transpose(1,2,0), -1, kernel) # H, W
        avg_filtered = avg_filtered[..., np.newaxis] # H, W, 1
        
        score = avg_filtered[..., 0].max()
        coord = np.nonzero(avg_filtered[..., 0] == score) # 2, n (y,x)
        coord_final = np.asarray(coord).transpose(1,0)[..., ::-1] # n, 2 (x,y)
        
        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[0] >= x_min and cord_list[0] <= x_max and
                    cord_list[1] >= y_min and cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break
        
        avg_filtered = torch.from_numpy(avg_filtered[..., 0]).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered + select_output[0].unsqueeze(-1))
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        _, valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3 
        
        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), save_path, coord_final,
                    img_ann[positives[k]]['bboxes'])
    return acc_num


def evaluate(feat_dir, output_path, decoder_ckpt_path, json_folder, mask_thresh, logger, camlist, dataset, pipeline, gaussians, background, encoder_mode):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    
    if dataset.speedup:
        feature_out_dim=camlist[0].img_embed.shape[1] # 512
        feature_in_dim = int(feature_out_dim/32)
        cnn_decoder = CNN_decoder(feature_in_dim, feature_out_dim)
        cnn_decoder_ckpt=torch.load(decoder_ckpt_path)
        if 'module_state_dict' in cnn_decoder_ckpt:
            cnn_decoder.load_state_dict(cnn_decoder_ckpt['module_state_dict'])
        else:
            cnn_decoder.load_state_dict(cnn_decoder_ckpt)
        
    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_folder), Path(output_path)) # eval image infos
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    compressed_sem_feats = torch.zeros(len(feat_dir), len(eval_index_list), *image_shape, feature_in_dim)
    for i in range(len(feat_dir)):

        for j, idx in enumerate(eval_index_list):
            viewcam = camlist[idx]
            viewcam.image_height=image_shape[0]
            viewcam.image_width=image_shape[1]
            render_pkg = render(viewcam, gaussians, pipeline, background)
            feature_map = render_pkg["render"] # 16,731,989  
            compressed_sem_feats[i][j] = feature_map.permute(1,2,0) # 1, num_eval_imgs, h, w, c=16

    # instantiate autoencoder and openclip
    if encoder_mode == 'default':
        clip_model = OpenCLIPNetwork(device)
    else:
        assert False, "encoder_mode not supported"

    iou_all = []
    acc_num = 0
    for j, idx in enumerate(tqdm(eval_index_list)): # 逐eval图处理
        image_name = Path(output_path) / f'{idx+1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)
        
        sem_feat = compressed_sem_feats[:, j, ...] # 1, h, w, c=16
        sem_feat = sem_feat.float().to(device)
        rgb_img = cv2.imread(image_paths[j])[..., ::-1] # BGR->RGB h, w, c=3
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            restored_feat = cnn_decoder(sem_feat.squeeze(0).permute(2,0,1)) # 512, h, w
            restored_feat = restored_feat.permute(1,2,0).unsqueeze(0)         # 1, h, w, 512
        
        img_ann = gt_ann[f'{idx}']
        clip_model.set_positives(list(img_ann.keys()))
        
        iou_list = activate_stream(restored_feat, rgb_img, clip_model, image_name, img_ann,
                                            thresh=mask_thresh, colormap_options=colormap_options)
        iou_all.extend(iou_list)
        # chosen_lvl_list.extend(c_lvl)

        acc_num_img = lerf_localization(restored_feat, rgb_img, clip_model, image_name, img_ann)
        acc_num += acc_num_img
        logger.info(f"eval: {idx+1:0>5} acc_num: {acc_num_img}/{len(list(img_ann.keys()))} mean_iou: {sum(iou_list)/len(iou_list):.4f}")

        torch.cuda.empty_cache()
    # # iou
    mean_iou_chosen = sum(iou_all) / len(iou_all)
    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")

    # localization acc
    total_bboxes = 0
    for img_ann in gt_ann.values():
        total_bboxes += len(list(img_ann.keys()))
    acc = acc_num / total_bboxes
    logger.info("Localization accuracy: " + f'{acc:.4f}')


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


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    parser = ArgumentParser(description="prompt any label")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--json_folder", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    parser.add_argument("--iteration", default=-1, type=int) 
    parser.add_argument('--encoder_mode', type=str, default='default') 
    args = get_combined_args(parser)
    print(args)

    # NOTE config setting
    dataset_name = args.dataset_name
    mask_thresh = args.mask_thresh
    encoder_mode = args.encoder_mode
    feat_dir = [os.path.join(args.model_path, 'train', "ours_{}".format(args.iteration), "feature_map_npy")]
    output_path = os.path.join(args.model_path, 'train', "ours_{}".format(args.iteration), "eval")
    json_folder = os.path.join(args.json_folder, dataset_name)
    decoder_ckpt_path = os.path.join(args.model_path, "decoder_chkpnt{}.pth".format(args.iteration))
    
    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)
    
    # NOTE load GS scene
    with torch.no_grad():
        
        dataset=model.extract(args)
        pipline=pipeline.extract(args)
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False) # load GS scene from *.ply file
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        camlist=scene.getTrainCameras()

        torch.cuda.empty_cache()
        
        # NOTE evaluate
        evaluate(feat_dir, output_path, decoder_ckpt_path, json_folder, mask_thresh, logger, camlist, dataset, pipline, gaussians, background, encoder_mode)