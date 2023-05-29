"""data loader."""

import random
import numpy as np
from torch.utils import data
from skimage.morphology import erosion, dilation
from custom.utils.common_tools import *

class MyDataset(data.Dataset):
    def __init__(
            self,
            dst_list_file,
            crop_box,
            transforms
    ):
        self.data_lst = self._load_files(dst_list_file)
        self._crop_box= crop_box
        self._transforms = transforms

    def _load_files(self, file):
        data_list = []
        with open(file, 'r') as f:
            for line in f:
                data_list.append(line.strip())
        return data_list

    def __getitem__(self, idx):
        source_data = self._load_source_data(self.data_lst[idx])
        return source_data

    def __len__(self):
        return len(self.data_lst)

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        org_img = data['img']
        org_mask = data['mask']
        cropped_img, cropped_mask = self.center_crop(org_img, org_mask)
        # transform前，数据必须转化为[C,H,D,W]的形状
        if self._transforms:
            cropped_img, cropped_mask = self._transforms(cropped_img, cropped_mask)
        return cropped_img, cropped_mask

    def center_crop(self, img, mask):
        img_patch =  img[:,self._crop_box[0]:self._crop_box[1],
                            self._crop_box[2]:self._crop_box[3],
                            self._crop_box[4]:self._crop_box[5]]
        mask_patch =  mask[:, self._crop_box[0]:self._crop_box[1],
                            self._crop_box[2]:self._crop_box[3],
                            self._crop_box[4]:self._crop_box[5]]
        return img_patch, mask_patch

