import argparse
import glob
import os
import sys
import tarfile
import traceback
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from cal_metrics import Get_Metrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer.predictor import SegBrainTumorModel, SegBrainTumorPredictor


def parse_args():
    parser = argparse.ArgumentParser(description='Test SegBrainTumor')

    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--input_path', default='../example/data/input', type=str)
    parser.add_argument('--output_path', default='../example/data/output', type=str)
    parser.add_argument(
        '--model_path',
        default=glob.glob("./data/model/*.tar")[0] if len(glob.glob("./data/model/*.tar")) > 0 else None,
        type=str,
    )
    parser.add_argument(
        '--model_file',
        type=str,
        default='../train/checkpoints/v1/60.pth'
        # default=None
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='./seg_braintumor.yaml'
        # default=None
    )
    args = parser.parse_args()
    return args


def inference(predictor: SegBrainTumorPredictor, volume: np.ndarray):
    pred_array = predictor.predict(volume)
    return pred_array


def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    sitk_img = reader.Execute()
    return sitk_img


def main(input_path, output_path, device, args):
    # TODO: 适配参数输入
    if (
        args.model_file is not None and 
        args.config_file is not None
    ):
        model_seg_braintumor = SegBrainTumorModel(
            model_f=args.model_file,
            config_f=args.config_file,
        )
        predictor_seg_braintumor = SegBrainTumorPredictor(
            device=device,
            model=model_seg_braintumor,
        )
    else:
        with tarfile.open(args.model_path, 'r') as tar:
            files = tar.getnames()
            model_seg_braintumor = SegBrainTumorModel(
                model_f=tar.extractfile(tar.getmember('seg_braintumor.pt')),
                config_f=tar.extractfile(tar.getmember('seg_braintumor.yaml')),
            )
            predictor_seg_braintumor = SegBrainTumorPredictor(
                device=device,
                model=model_seg_braintumor,
            )

    os.makedirs(output_path, exist_ok=True)

    name=["Sample Id",'WT Dice','WT Sensitivity','WT Specificity','TC Dice','TC Sensitivity','TC Specificity', 'ET Dice','ET Sensitivity','ET Specificity']
    result = []
    for pid in tqdm(os.listdir(input_path)):
        flair_itk = sitk.ReadImage(os.path.join(input_path, pid, pid+"_flair.nii.gz"))
        t1ce_itk = sitk.ReadImage(os.path.join(input_path, pid, pid+"_t1ce.nii.gz"))
        t1_itk = sitk.ReadImage(os.path.join(input_path, pid, pid+"_t1.nii.gz"))
        t2_itk = sitk.ReadImage(os.path.join(input_path, pid, pid+"_t2.nii.gz"))   
        flair = sitk.GetArrayFromImage(flair_itk)[np.newaxis,:,:,:]
        t1ce = sitk.GetArrayFromImage(t1ce_itk)[np.newaxis,:,:,:]
        t1 = sitk.GetArrayFromImage(t1_itk)[np.newaxis,:,:,:]
        t2 = sitk.GetArrayFromImage(t2_itk)[np.newaxis,:,:,:]
        volume = np.concatenate([flair,t1ce,t1,t2],0).astype('float32')

        pred_array = inference(predictor_seg_braintumor, volume)
        pred_itk = sitk.GetImageFromArray(pred_array)
        pred_itk.CopyInformation(flair_itk)
        sitk.WriteImage(pred_itk, os.path.join(output_path, f'{pid}_seg.nii.gz'))

        seg_itk = sitk.ReadImage(os.path.join(input_path, pid, pid+"_seg.nii.gz"))     
        seg = sitk.GetArrayFromImage(seg_itk)
        metics = Get_Metrics(pred_array, seg)
        sample = [pid] + list(metics.values())
        result.append(sample)
        log = pd.DataFrame(columns=name, data=result)
        log.to_csv(os.path.join(output_path,"result.csv"), encoding='utf-8', index=False)
    for i, item in enumerate(name[1:]):
        vals = [row[i+1] for row in result]
        _mean = sum(vals)/len(vals)
        print(item + ": ", _mean)

if __name__ == '__main__':
    args = parse_args()
    main(
        input_path=args.input_path,
        output_path=args.output_path,
        device=args.device,
        args=args,
    )