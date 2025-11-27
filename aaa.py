import os
import numpy as np
import torch


def extractGTs(gt, erodeKernSize=15, dilateKernSize=11):
    from scipy.ndimage.filters import minimum_filter, maximum_filter
    gt1 = minimum_filter(gt, erodeKernSize)
    gt0 = np.logical_not(maximum_filter(gt, dilateKernSize))
    return gt0, gt1


def computeMetricsContinue(values, gt0, gt1):

    values = values.flatten().cpu().numpy().astype(np.float32)
    gt0 = gt0.flatten().astype(np.float32)
    gt1 = gt1.flatten().astype(np.float32)


    inds = np.argsort(values)
    inds = inds[(gt0[inds] + gt1[inds]) > 0]
    vet_th = values[inds]
    gt0 = gt0[inds]
    gt1 = gt1[inds]

    TN = np.cumsum(gt0)
    FN = np.cumsum(gt1)
    FP = np.sum(gt0) - TN
    TP = np.sum(gt1) - FN

    msk = np.pad(vet_th[1:] > vet_th[:-1], (0, 1), mode='constant', constant_values=True)
    FP = FP[msk]
    TP = TP[msk]
    FN = FN[msk]
    TN = TN[msk]
    vet_th = vet_th[msk]

    return FP, TP, FN, TN, vet_th


# gt= torch.randint(0,2,(2,1,512,512))
gt = torch.zeros(2,1,512,512)
gt[:,:,100:300,200:400]=1
print(gt)
gt0,gt1 = extractGTs(gt)


# predict = torch.ran
FP, TP, FN, TN, vet_th = computeMetricsContinue(gt,gt0,gt1)

print(FN)