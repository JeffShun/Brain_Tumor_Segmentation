"""生成模型输入数据."""

import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./train_data/origin_data')
    parser.add_argument('--tgt_path', type=str, default='./train_data/processed_data')
    args = parser.parse_args()
    return args


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    img = reader.Execute()
    return img


def gen_lst(tgt_path, task, processed_pids):
    save_file = os.path.join(tgt_path, task+'.lst')
    data_list = glob.glob(os.path.join(tgt_path, '*.npz'))
    num = 0
    with open(save_file, 'w') as f:
        for pid in processed_pids:
            data = os.path.join(tgt_path, pid+".npz")
            if data in data_list:
                num+=1
                f.writelines(data + '\r\n')
    print('num of data: ', num)


def process_single(input):
    seg_path, flair_path, t1ce_path, t1_path, t2_path, tgt_path, pid = input
    flair_itk = sitk.ReadImage(flair_path)
    t1ce_itk = sitk.ReadImage(t1ce_path)
    t1_itk = sitk.ReadImage(t1_path)
    t2_itk = sitk.ReadImage(t2_path)
    seg_itk = sitk.ReadImage(seg_path)

    flair = sitk.GetArrayFromImage(flair_itk)[np.newaxis,:,:,:]
    t1ce = sitk.GetArrayFromImage(t1ce_itk)[np.newaxis,:,:,:]
    t1 = sitk.GetArrayFromImage(t1_itk)[np.newaxis,:,:,:]
    t2 = sitk.GetArrayFromImage(t2_itk)[np.newaxis,:,:,:]
    img = np.concatenate([flair,t1ce,t1,t2],0)

    seg = sitk.GetArrayFromImage(seg_itk)
    wt = (seg>0).astype("uint8")[np.newaxis,:,:,:]
    tc = ((seg==1)|(seg==4)).astype("uint8")[np.newaxis,:,:,:]
    et = (seg==4).astype("uint8")[np.newaxis,:,:,:]
    mask = np.concatenate([wt,tc,et],0)
    np.savez_compressed(os.path.join(tgt_path, f'{pid}.npz'), img=img, mask=mask)



if __name__ == '__main__':
    args = parse_args()
    src_path = args.src_path
    for task in ["train", "valid"]:
        print("\nBegin gen %s data!"%(task))
        src_data_path = os.path.join(args.src_path, task)
        tgt_path = args.tgt_path
        os.makedirs(tgt_path, exist_ok=True)
        inputs = []
        processed_pids = os.listdir(src_data_path)
        for pid in tqdm(processed_pids):
            seg_path = os.path.join(src_data_path, pid, pid+"_seg.nii.gz")           
            flair_path = os.path.join(src_data_path, pid, pid+"_flair.nii.gz")
            t1ce_path = os.path.join(src_data_path, pid, pid+"_t1ce.nii.gz")
            t1_path = os.path.join(src_data_path, pid, pid+"_t1.nii.gz")
            t2_path = os.path.join(src_data_path, pid, pid+"_t2.nii.gz")
            inputs.append([seg_path, flair_path, t1ce_path, t1_path, t2_path, tgt_path, pid])
        pool = Pool(8)
        pool.map(process_single, inputs)
        pool.close()
        pool.join()
        # 生成Dataset所需的数据列表
        gen_lst(tgt_path, task, processed_pids)
