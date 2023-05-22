from calendar import c
from dis import dis
import os
import sys
from os.path import abspath, dirname
from typing import IO, Dict
import SimpleITK as sitk

import numpy as np
import torch
import yaml
import random
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import dilation
from train.config.seg_braintumor_config import network_cfg


class SegBrainTumorConfig:

    def __init__(self, test_cfg):
        # 配置文件
        self.patch_size = test_cfg.get('patch_size')
        self.seg_thresh = test_cfg.get('seg_thresh')
    def __repr__(self) -> str:
        return str(self.__dict__)


class SegBrainTumorModel:

    def __init__(self, model_f: IO, config_f):
        # TODO: 模型文件定制
        self.model_f = model_f 
        self.config_f = config_f
        self.network_cfg = network_cfg


class SegBrainTumorPredictor:

    def __init__(self, device: str, model: SegBrainTumorModel):
        self.device = torch.device(device)
        self.model = model

        with open(self.model.config_f, 'r') as config_f:
            self.test_cfg = SegBrainTumorConfig(yaml.safe_load(config_f))
        self.network_cfg = model.network_cfg
        self.load_model()

    def load_model(self) -> None:
        if isinstance(self.model.model_f, str):
            # 根据后缀判断类型
            if self.model.model_f.endswith('.pth'):
                self.load_model_pth()
            else:
                self.load_model_jit()

    def load_model_jit(self) -> None:
        # 加载静态图
        from torch import jit
        self.net = jit.load(self.model.model_f, map_location=self.device)
        self.net.eval()
        self.net.to(self.device).half()

    def load_model_pth(self) -> None:
        # 加载动态图
        self.net = self.network_cfg.network
        checkpoint = torch.load(self.model.model_f, map_location=self.device)
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        self.net.to(self.device).half()

    def _get_bbox(self, data, border_pad):
        shape = data.shape[1:]
        loc = np.where(data.sum(0) > 0)
        zmin = max(0, np.min(loc[0]) - border_pad[0])
        zmax = min(shape[0], np.max(loc[0]) + border_pad[0]) + 1
        ymin = max(0, np.min(loc[1]) - border_pad[1])
        ymax = min(shape[1], np.max(loc[1]) + border_pad[1]) + 1
        xmin = max(0, np.min(loc[2]) - border_pad[2])
        xmax = min(shape[2], np.max(loc[2]) + border_pad[2]) + 1
        return zmin, zmax, ymin, ymax, xmin, xmax

    def predict(self, volume: np.ndarray):
        bbox = self._get_bbox(volume, (2, 2, 2))
        volume_crop = volume[:,bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]: bbox[5]]
        seg_pred  = self._forward(volume_crop)
        res_pred = np.zeros(volume[0,:,:,:].shape, dtype='uint8')
        res_pred[bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]: bbox[5]] = seg_pred
        return res_pred

    def _forward(self, volume: np.ndarray):
        shape = volume.shape[1:]
        volume = torch.from_numpy(volume).float()
        volume = self._normlize(volume)[None]
        volume = self._resize_torch(volume, self.test_cfg.patch_size)

        with torch.no_grad():
            patch_gpu = volume.half().to(self.device)
            seg_heatmap = self.net.forward_test(patch_gpu)
            seg_heatmap = torch.sigmoid(seg_heatmap)
            seg_heatmap = self._resize_torch(seg_heatmap, shape)    
            seg_arr = seg_heatmap.squeeze().cpu().detach().numpy()
            out_mask = np.zeros_like(seg_arr[0,:,:,:])
            wt = (seg_arr[0,:,:,:] > self.test_cfg.seg_thresh).astype(np.uint8)
            tc = (seg_arr[1,:,:,:] > self.test_cfg.seg_thresh).astype(np.uint8)
            et = (seg_arr[2,:,:,:] > self.test_cfg.seg_thresh).astype(np.uint8)
            out_mask[wt==1] = 2
            out_mask[(tc==1) & (wt==1)] = 1
            out_mask[(et==1) & (tc==1) & (wt==1)] = 4

        return out_mask.astype(np.uint8)

    def _normlize(self, img, win_clip=None):
        ori_shape = img.shape
        img_o = img.reshape(ori_shape[0], -1)
        if win_clip is not None:
            img_o = torch.clip(img_o, win_clip[0], win_clip[1])
        img_min = img_o.min(dim=-1,keepdim=True)[0]
        img_max = img_o.max(dim=-1,keepdim=True)[0]
        img_o = (img_o - img_min)/(img_max - img_min)
        img_o = img_o.reshape(ori_shape)
        return img_o 

    def _resize_torch(self, data, scale, mode="trilinear"):
        return torch.nn.functional.interpolate(data, size=scale, mode=mode)    