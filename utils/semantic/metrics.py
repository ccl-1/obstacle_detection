# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Model validation metrics."""

import numpy as np
import torch

import torch
from torch import Tensor
from ..metrics import ap_per_class
import torch.nn.functional as F


    # final_metric = mp_bbox, mr_bbox, map50_bbox, map_bbox, pa_sem, mpa_sem, miou_sem, fwiou_sem # 8, len(loss)=5 
#
def fitness(x):
    # Model fitness as a weighted combination of metrics
    w1 = [0.0, 0.0, 0.9, 0.9]   # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    w2 = [0.1, 0.9, 0.9, 0.1]   # pa, mpa , MIOU,FWMIOU]
    return (x[:, :4] * w1).sum(1) + (x[:, -4:] * w2).sum(1)
    

def ap_per_class_box(
        tp_b,
        conf,
        pred_cls,
        target_cls,
        plot=False,
        save_dir=".",
        names=(),
):
    """
    Args:
        tp_b: tp of boxes.
        tp_m: tp of masks.
        other arguments see `func: ap_per_class`.
    """
    results_boxes = ap_per_class(tp_b,
                                 conf,
                                 pred_cls,
                                 target_cls,
                                 plot=plot,
                                 save_dir=save_dir,
                                 names=names,
                                 prefix="Box")[2:]
    # return tp, fp, p, r, f1, ap, unique_classes.astype(int)[2:]

    results = {
        "boxes": {
            "p": results_boxes[0],
            "r": results_boxes[1],
            "ap": results_boxes[3],
            "f1": results_boxes[2],
            "ap_class": results_boxes[4]}}
    return results

class Metric:
    def __init__(self) -> None:
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )

    @property
    def ap50(self):
        """
        AP@0.5 of all classes.

        Return:
            (nc, ) or [].
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """AP@0.5:0.95
        Return:
            (nc, ) or [].
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Mean precision of all classes.

        Return:
            float.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Mean recall of all classes.

        Return:
            float.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        Mean AP@0.5 of all classes.

        Return:
            float.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Mean AP@0.5:0.95 of all classes.

        Return:
            float.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map."""
        return (self.mp, self.mr, self.map50, self.map)

    def class_result(self, i):
        """Class-aware result, return p[i], r[i], ap50[i], ap[i]"""
        return (self.p[i], self.r[i], self.ap50[i], self.ap[i])

    def get_maps(self, nc):
        """Calculates and returns mean Average Precision (mAP) for each class given number of classes `nc`."""
        maps = np.zeros(nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def update(self, results):
        """
        Args:
            results: tuple(p, r, ap, f1, ap_class)
        """
        p, r, all_ap, f1, ap_class_index = results
        self.p = p
        self.r = r
        self.all_ap = all_ap
        self.f1 = f1
        self.ap_class_index = ap_class_index


class Metrics:
    """Metric for boxes."""

    def __init__(self) -> None:
        self.metric_box = Metric()

    def update(self, results):
        """
        Args:
            results: Dict{'boxes': Dict{}
        """
        self.metric_box.update(list(results["boxes"].values()))

    def mean_results(self):
        """Computes and returns the mean results for both box and mask metrics by summing their individual means."""
        return self.metric_box.mean_results() 

    def class_result(self, i):
        """Returns the sum of box and mask metric results for a specified class index `i`."""
        return self.metric_box.class_result(i)

    def get_maps(self, nc):
        """Calculates and returns the sum of mean average precisions (mAPs) for both box and mask metrics for `nc`
        classes.
        """
        return self.metric_box.get_maps(nc)

    @property
    def ap_class_index(self):
        """Returns the class index for average precision, shared by both box and mask metrics."""
        return self.metric_box.ap_class_index


class SegmentationMetric(object):
    '''
    imgLabel [batch_size, height(144), width(256)]
    confusionMatrix [[0(TN),1(FP)],
                     [2(FN),3(TP)]]
    '''
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
        
    def lineAccuracy(self):
        Acc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=1) + 1e-12)
        return Acc[1]

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / (self.confusionMatrix.sum(axis=0) + 1e-12)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union 
        IoU[np.isnan(IoU)] = 0 #  返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)
        return mIoU
    
    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        IoU = intersection / union
        IoU[np.isnan(IoU)] = 0
        return IoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        # print(imgLabel.shape)
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix
    
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


    def addBatch(self, imgPredict, imgLabel):
        "https://blog.csdn.net/m0_47355331/article/details/119972157  addBatch(self, imgPredict, imgLabel, ignore_labels)"
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1): # n = batch size
        # out 是batch内计算的平均结果，到当前 batch之前的所有数据的验证结果 ... 
        self.val = val 
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count if self.count != 0 else 0 # 平均精度


class Semantic_Metrics:
    def __init__(self, nc):
        self.device = 'cpu'
        self.metric_mask = SegmentationMetric(nc)

    def update(self, pred_masks, gt_masks): # (b,c,h,w), (b,h,w)
        pred_masks = pred_masks.to(self.device)
        gt_masks = gt_masks.to(self.device)

        semantic_masks = []
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            size = gt_mask.size()[-2:]
            mask_size = pred_mask.size()[-2:]
            if  size != mask_size:
                pred_mask = F.interpolate(pred_mask, size=size, mode='bilinear', align_corners=True)
            pred_mask = torch.squeeze(pred_mask)        # (b,c,w,h) ->  (c,w,h)
            pred_mask = torch.argmax(pred_mask, dim=0)  # (c,h,w) -> (h,w)
            semantic_masks.append(pred_mask) 
        pred_masks = torch.stack(semantic_masks, dim=0) # (b,w,h)
        
        self.metric_mask.reset()
        self.metric_mask.addBatch(pred_masks, gt_masks)
        
        cpa = self.metric_mask.classPixelAccuracy()
        pa = self.metric_mask.pixelAccuracy()
        mpa = self.metric_mask.meanPixelAccuracy()

        IoU = self.metric_mask.IntersectionOverUnion()
        mIoU = self.metric_mask.meanIntersectionOverUnion()
        FWIoU = self.metric_mask.Frequency_Weighted_Intersection_over_Union()

        return cpa, pa, mpa, IoU, mIoU, FWIoU


    

KEYS = [
    "train/box_loss",           # train loss
    "train/obj_loss",
    "train/cls_loss",
    "train/seg_loss", 
    "train/dic_loss",
    "metrics/precision(B)",     # metrics
    "metrics/recall(B)",
    "metrics/mAP_0.5(B)",
    "metrics/mAP_0.5:0.95(B)", 
    # "metrics/CPA(S)",
    # "metrics/IoU(S)",       
    "metrics/PA(S)",       
    "metrics/MPA(S)",       
    "metrics/MIOUS(S)",        
    "metrics/FWIOUS(S)",       
    "val/box_loss",             # val loss
    "val/obj_loss",
    "val/cls_loss",
    "val/seg_loss", 
    "val/dic_loss",
    "x/lr0",                    # lr
    "x/lr1",
    "x/lr2",
]


BEST_KEYS = [
    "best/epoch",
    "best/precision(B)",
    "best/recall(B)",
    "best/mAP_0.5(B)",
    "best/mAP_0.5:0.95(B)",
    "best/MPA(S)",       
    "best/MIOUS(S)",
    "best/FWIOUS(S)",]
