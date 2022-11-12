from pathlib import Path

import numpy as np
import torch

import nibabelio
import path_explorer as px

def iou(pred, segm):
    intersection = np.sum(pred * segm) + 1
    union = np.sum((pred + segm).clip(0, 1)) + 1
    return float(intersection/union)

def scores(pred, segm):
    return (
        iou(pred > 0, segm > 0),
        iou(pred == 1, segm == 1),
        iou(pred == 2, segm == 2)
    )

if __name__=="__main__":
    sources = Path("../sources")
    prediction_name = "nnunet_prediction.nii.gz"
    count, both, liver, tumor = 0, 0, 0, 0
    for case_name in px.iter_trainable(sources):
        _, bottom, top, height = nibabelio.load_registration_data(sources/case_name)
        data = nibabelio.load(sources/case_name, scan=False, train=True)
        segm = data["segm"]
        pred = nibabelio._load_ndarray(sources/case_name/prediction_name)
        assert pred.shape[-1] == height
        pred = pred[..., bottom:top]
        b, l, t = scores(pred, segm)
        print(case_name, f"both: {100*b:.0f}", f"liver: {100*l:.0f}", f"tumor: {100*t:.0f}")
        count += 1
        both += b
        liver += l
        tumor += t
    print("General", f"both: {100*both/count:.0f}", f"liver: {100*liver/count:.0f}", f"tumor: {100*tumor/count:.0f}")