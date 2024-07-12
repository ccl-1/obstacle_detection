# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Validate a trained YOLOv5 segment model on a segment dataset.

Usage:
    $ bash data/scripts/get_coco.sh --val --segments  # download COCO-segments val split (1G, 5000 images)
    $ python segment/val.py --weights yolov5s-seg.pt --data coco.yaml --img 640  # validate COCO-segments

Usage - formats:
    $ python segment/val.py --weights yolov5s-seg.pt                 # PyTorch
                                      yolov5s-seg.torchscript        # TorchScript
                                      yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s-seg_openvino_label     # OpenVINO
                                      yolov5s-seg.engine             # TensorRT
                                      yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                      yolov5s-seg_saved_model        # TensorFlow SavedModel
                                      yolov5s-seg.pb                 # TensorFlow GraphDef
                                      yolov5s-seg.tflite             # TensorFlow Lite
                                      yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                      yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import json
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path
from pycocotools import mask as maskUtils

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml

from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch.nn.functional as F

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.general import (
    intersect_dicts,
    LOGGER,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    bdd100k_10_to_5_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
)

from utils.metrics import ConfusionMatrix, box_iou, ap_per_class
from utils.plots import output_to_target, plot_val_study, plot_images

from utils.semantic.plots import plot_images_and_masks

from utils.semantic.dataloaders import create_dataloader 
from utils.semantic.metrics import Semantic_Metrics, Metrics, ap_per_class_box, AverageMeter
from utils.torch_utils import select_device, smart_inference_mode




not_ignore_ids = [1]
all_stuff_ids = [1,
    183, # other
    0, # unlabeled
]

label_mapping_object = {0: 0, 255: 1} 
label_mapping_obstacle = {0: 0, 215: 1} # ignore_label=-1,   for obstacle
label_mapping_bdd100k = {0: 0, 255:1, 127:2} # ignore_label=-1,    for bdd 100k  0, 127, 255
label_mapping_rs19 = {0: 0, 255: 1}
label_mapping_coco128 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 
                         12: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 21: 17, 22: 18, 23: 19, 
                         24: 20, 25: 21, 26: 22, 27: 23, 28: 24, 29: 25, 30: 26, 31: 27, 32: 28, 
                         33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 39: 34, 40: 35, 41: 36, 42: 37, 
                         43: 38, 44: 39, 45: 40, 46: 41, 47: 42, 49: 43, 50: 44, 51: 45, 52: 46, 
                         53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 
                         62: 56, 63: 57, 64: 58, 65: 59, 66: 60, 68: 61, 69: 62, 70: 63, 72: 64, 
                         73: 65, 74: 66, 75: 67, 76: 68, 77: 69, 78: 70, 80: 71}

def getDataIds(name = 'semantic'):
    if 'semantic' == name:
        return  all_stuff_ids
    else:
        print("Not defined format.")

    
def save_one_txt(predn, save_conf, shape, file):
    """Saves detection results in txt format; includes class, xywh (normalized), optionally confidence if `save_conf` is
    True.
    """
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map, pred_masks):
    """
    Saves a JSON file with detection results including bounding boxes, category IDs, scores, and segmentation masks.

    Example JSON result: {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}.
    """
    from pycocotools.mask import encode

    def single_encode(x):
        rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    pred_masks = np.transpose(pred_masks, (2, 0, 1))
    with ThreadPool(NUM_THREADS) as pool:
        rles = pool.map(single_encode, pred_masks)
    for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
                "segmentation": rles[i],
            }
        )

def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix.

    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    max_det=300,  # maximum detections per image
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val-seg",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    label_mapping = None,
    use_bdd100k_5 = True,
    data_mode=None,

    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    overlap=False,
    mask_downsample_ratio=1,
    compute_loss=None,
    callbacks=Callbacks(),
):
    if label_mapping == 'bdd100k':
        label_map = label_mapping_bdd100k
    elif label_mapping == 'coco128':
        label_map = label_mapping_coco128
    elif label_mapping == 'rs19':
        label_map = label_mapping_rs19
    elif label_mapping == 'object':
        label_map = label_mapping_object
    else: # label_map == 'obstacle':
        label_map = label_mapping_obstacle

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()        
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        # nm = de_parallel(model).model.model[-1].nm if isinstance(model, SegmentationModel) else 32  # number of masks
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != "cpu"
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(f"coco{os.sep}val2017.txt")  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    seg_names = data.get('seg_names', [])  # names of semantic classes
    seg_nc = len(seg_names)  # number of stuff classes

    if use_bdd100k_5:
        names = {0: 'car', 1: 'person', 2: 'rider', 3: 'traffic sign', 4: 'traffic light'}
        nc = 5
        data["nc"] = 5
   

    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Semantic Segmentation
    img_id_list = []

    # Dataloader
    if not training:
        if not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            # print("============\n", ncm, nc)
            assert ncm == nc, (
                f"{weights} ({ncm} classes) trained on different --data than what you passed ({nc} "
                f"classes). Pass correct combination of --weights and --data that are trained together."
            )
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad, rect = (0.0, False) if task == "speed" else (0.5, pt)  # square inference for benchmarks
        task = task if task in ("train", "val", "test") else "val"  # path to train/val/test images
        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            pad=pad,
            rect=rect,
            workers=workers,
            prefix=colorstr(f"{task}: "),
            overlap_mask=overlap,
            mask_downsample_ratio=mask_downsample_ratio,
            label_mapping=label_map,
            use_bdd100k_5 = use_bdd100k_5,
            data_mode=data_mode,
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, "names") else model.module.names  # get class names
    if isinstance(names, (list, tuple)):  # old format
        names = dict(enumerate(names))
    # class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    class_map = bdd100k_10_to_5_class()  if use_bdd100k_5  else list(range(1000))
    s = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)",)

    dt = Profile(device=device), Profile(device=device), Profile(device=device)

    metrics = Metrics()
    cpa_seg = AverageMeter()
    pa_seg = AverageMeter()
    mpa_seg = AverageMeter()
    IoU_seg = AverageMeter()
    mIoU_seg = AverageMeter()
    FWIoU_seg = AverageMeter()

    
    loss = torch.zeros(5, device=device)
    jdict, stats = [], []
    pbar = tqdm(dataloader, desc=s, bar_format=TQDM_BAR_FORMAT)  # progress bar # logger title head here  ... 
    for batch_i, (im, targets, gt_masks, paths, shapes) in enumerate(pbar):
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                targets = targets.to(device)
                gt_masks = gt_masks.to(device)
            gt_masks = gt_masks.float()
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width
        
        # from torchvision.utils import save_image
        # save_image(im[0], 'test_im.png')

        # Inference
        with dt[1]:
            output = model(im) if compute_loss else model(im, augment=augment)
            preds, train_out, pred_masks = output[0][0],  output[0][1], output[1]
            # torch.Size([b, 25200, 31]) , list(3).size=orch.Size([b, 3, 84, 84, 31]),    torch.Size([b, 2, 640, 640])  
        
        # Loss
        if compute_loss:
            loss += compute_loss((train_out, pred_masks), targets, gt_masks)[1]  # box, obj, cls
        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with dt[2]:
            # print( preds.size()) torch.Size([32, 27783, 31]) -> list[0]=[300, 6]
            preds = non_max_suppression(   # return list of detections, on (batch, 6) tensor per image [xyxy, conf, cls]
                preds, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls, max_det=max_det)

        # Metrics
        semantic_metrics = Semantic_Metrics(nc = seg_nc)
        cpa, pa, mpa, IoU, mIoU, FWIoU = semantic_metrics.update(pred_masks, gt_masks) # # (b,c,h,w), (b,h,w)
        batch = pred_masks.size()[0]
        cpa_seg.update(cpa, batch)
        pa_seg.update(pa, batch)
        mpa_seg.update(mpa, batch)
        IoU_seg.update(IoU, batch)
        mIoU_seg.update(mIoU, batch)
        FWIoU_seg.update(FWIoU, batch)


        semantic_masks = [] # Store pred_masks, which are converted to single channels in batch_i
        for si, (pred,  pred_mask) in enumerate(zip(preds, pred_masks)): # 遍历 batch 内的图片
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            image_id = path.stem # #文件名不带后缀
            img_id_list.append(image_id)

            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv) 
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

           # Save/log
            if save_txt:
                (save_dir / "labels").mkdir(parents=True, exist_ok=True)
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary

            # --------------------  Semantic Segmentation ------------------------------------
            # h0, w0 = shape
            h0, w0 = gt_masks.shape[-2:]

            _, mask_h, mask_w = pred_mask.shape # # ([2, 672, 672])
            h_ratio = mask_h / h0
            w_ratio = mask_w / w0

            if h_ratio == w_ratio: 
                pred_mask = F.interpolate(pred_mask[None], size = (h0, w0), mode = 'bilinear', align_corners = False)
            else:
                transform = transforms.CenterCrop((h0, w0))
                if (1 != h_ratio) and (1 != w_ratio):
                    h_new = h0 if (h_ratio < w_ratio) else int(mask_h / w_ratio)
                    w_new = w0 if (h_ratio > w_ratio) else int(mask_w / h_ratio)
                    pred_mask = F.interpolate(pred_mask[None], size = (h_new, w_new), mode = 'bilinear', align_corners = False)
                pred_mask = transform(pred_mask)

            # pred
            pred_mask = torch.squeeze(pred_mask) # 1xcxhxw ->  cxhxw
            seg_nc, h, w = pred_mask.shape
            pred_mask = torch.argmax(pred_mask, dim=0) # torch.Size([640, 640]) 返回指定维度最大值的序号, (c,h,w) -> (h,w)
            semantic_masks.append(pred_mask) # list(hw,hw)
        
        plot_semasks = []  # masks for plotting
        semantic_masks = torch.stack(semantic_masks, dim=0) # (b,w,h)
        if plots and batch_i < 3:
            plot_semasks.append(semantic_masks.clone().detach().cpu())

        # Plot images
        if plots and batch_i < 3:
            if len(plot_semasks):
                plot_semasks = torch.cat(plot_semasks, dim=0)
            plot_images_and_masks(im, targets, gt_masks, seg_nc, paths, save_dir / f"val_batch{batch_i}_labels.jpg", names)
            plot_images_and_masks(im, output_to_target(preds, max_det=15),
                plot_semasks, seg_nc, paths,save_dir / f"val_batch{batch_i}_pred.jpg", names,)  # pred

    # Compute metrics for detection
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        results = ap_per_class_box(*stats, plot=plots, save_dir=save_dir, names=names)
        metrics.update(results)
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

     # Compute metrics for semantic segmentation （pixelAccuracy， IntersectionOverUnion， meanIntersectionOverUnion）
    semantic_result = ( pa_seg.avg, mpa_seg.avg,  mIoU_seg.avg, FWIoU_seg.avg)

    # Print results of detection and semantic
    LOGGER.info(("%22s" + "%11s" * 5) % ("Class", "Images",'Mask(PA', 'mPA', 'MIoU', 'FWIoU)'))
    pf = '%22s' + '%11i' * 1 + '%11.3g' * 4  # print format 5
    LOGGER.info(pf % ("all_mask", seen, *semantic_result))

    LOGGER.info(("%22s" + "%11s" * 3) % ("Class", "Images",'Mask(CPA', 'IoU'))
    pf = '%22s' + '%11i' * 1 + '%11.3g' * 2  # print format 5
    
    for idx, (cpa, iou) in enumerate(zip( cpa_seg.avg, IoU_seg.avg)):
        LOGGER.info(pf % (str(idx), seen, cpa, iou))
    LOGGER.info(pf % ('avg', seen, np.mean(cpa_seg.avg[1:]), np.mean(IoU_seg.avg[1:])))
    

    pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format 4
    LOGGER.info(pf % ("all_bbox", seen, nt.sum(), *metrics.mean_results()))


    if nt.sum() == 0:
        LOGGER.warning(f"WARNING ⚠️ no labels found in {task} set, can not compute metrics without labels")


    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(metrics.ap_class_index):
            LOGGER.info(pf % (names[c], seen, nt[c], *metrics.class_result(i)))

    # Print speeds
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}" % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    # callbacks.run('on_val_end')

    mp_bbox, mr_bbox, map50_bbox, map_bbox = metrics.mean_results()
    cpa_sem, iou_sem = np.mean(cpa_seg.avg[1:]) , np.mean(IoU_seg.avg[1:])
    pa_sem, mpa_sem, miou_sem, fwiou_sem = semantic_result

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ""  # weights
        anno_json = str(Path("../datasets/coco/annotations/instances_val2017.json"))  # annotations
        if not os.path.exists(anno_json):
            anno_json = os.path.join(data["path"], "annotations", "instances_val2017.json")
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements("pycocotools>=2.0.6")
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}") 
    final_metric = mp_bbox, mr_bbox, map50_bbox, map_bbox, pa_sem, cpa_sem, iou_sem, fwiou_sem # 8, len(loss)=5 
    return (*final_metric, *(loss.cpu() / len(dataloader)).tolist()),     metrics.get_maps(nc), t # val loss
    # results = (*final_metric, *(loss.cpu() / len(dataloader)).tolist())



def parse_opt():
    """Parses command line arguments for configuring YOLOv5 options like dataset path, weights, batch size, and
    inference settings.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/bdd100k-seg.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default="runs/train-seg/bdd100k_det_first/exp/weights/last.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")

    parser.add_argument("--label_mapping", type=str, default="obstacle", help="dataset.yaml path")
    parser.add_argument("--use_bdd100k_5", type=bool, default=False, help=" use_bdd100k_5 or not ")
    parser.add_argument("--data_mode", type=str, default=None, help="det, seg") 

    parser.add_argument("--conf-thres", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--single-cls", action="store_true", help="treat as single-class dataset")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-hybrid", action="store_true", help="save label+prediction hybrid results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-json", action="store_true", help="save a COCO-JSON results file")
    parser.add_argument("--project", default=ROOT / "runs/val-seg", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    # opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 tasks including training, validation, testing, speed, and study with configurable options."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.warning(f"WARNING ⚠️ confidence threshold {opt.conf_thres} > 0.001 produces invalid results")
        if opt.save_hybrid:
            LOGGER.warning("WARNING ⚠️ --save-hybrid returns high mAP from hybrid labels, not from predictions alone")
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = torch.cuda.is_available() and opt.device != "cpu"  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            subprocess.run(["zip", "-r", "study.zip", "study_*.txt"])
            plot_val_study(x=x)  # plot
        else:
            raise NotImplementedError(f'--task {opt.task} not in ("train", "val", "test", "speed", "study")')


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
