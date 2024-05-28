import contextlib
import argparse
import os
import sys
import cv2
from shapely.geometry import Polygon, box
from shapely.affinity import translate
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import  check_img_size,check_requirements,print_args
from utils.torch_utils import select_device, smart_inference_mode
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    cv2,
    increment_path,
    scale_boxes,
    non_max_suppression,
    print_args,
)


def extend_track_region(track_polygon, extension_length):
    minx, miny, maxx, maxy = track_polygon.bounds
    extended_region = track_polygon.union(
        translate(track_polygon, xoff=-extension_length) |
        translate(track_polygon, xoff=extension_length)
    )
    return extended_region

def calculate_danger_level(detections, extended_track_polygon, low_threshold, high_threshold):
    danger_levels = []
    for detection in detections:
        det_dict = {}
        print(detection[0].size())
        det_dict['bbox'] = detection[0, :4].cpu().numpy()
        det_dict['class'] = detection[0, 5].cpu().numpy()
        print(det_dict['class'])

        x1, y1, x2, y2 = detection[0, :4]
        object_box = box(x1, y1, x2, y2)
        if extended_track_polygon.intersects(object_box):
            intersection_area = extended_track_polygon.intersection(object_box).area
            object_area = object_box.area
            overlap_ratio = intersection_area / object_area

            if overlap_ratio > high_threshold:
                danger_level = "High"
            elif overlap_ratio > low_threshold:
                danger_level = "Medium"
            else:
                danger_level = "Low"
        else:
            danger_level = "Low"
        danger_levels.append({
            "object": det_dict,
            "danger_level": danger_level
        })
    return danger_levels


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s-seg.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/predict-seg",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False,
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.eval()
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            out = model(im, augment=augment, visualize=visualize) # 3  preds, protos, train_out 
            pred, train_out, pred_mask = out[0][0],  out[0][1], out[1][0]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        pred_mask = F.interpolate(pred_mask[None], size = im.shape[-2:], mode = 'nearest')
        pred_mask = torch.squeeze(pred_mask) # 1xcxhxw ->  cxhxw

        seg_nc, h, w = pred_mask.shape
        pred_mask = torch.argmax(pred_mask, dim=0) # torch.Size([640, 640]) 返回指定维度最大值的序号, (c,h,w) -> (h,w)
        segmentation_result = pred_mask.clone().detach().cpu().numpy() # 0 1 2 # (1200, 5760)
        segmentation_result = segmentation_result.astype(np.uint8)
        # _, segmentation_result = cv2.threshold(segmentation_result, 0, 255, cv2.THRESH_BINARY)
        
        # ===============================================================
        low_risk_th  = 0.1
        high_risk_th = 0.5
        extension_length = 50  

        
        # Morphological Operations
        kernel = np.ones((5, 5), np.uint8)
        closed_mask = cv2.morphologyEx(segmentation_result, cv2.MORPH_CLOSE, kernel)
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel) # remove noise
        contours, _ = cv2.findContours(segmentation_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        im_rgb = torch.squeeze(im, dim=0).permute(1,2,0).cpu().numpy().copy()
        cv2.drawContours(im_rgb, contours, -1, (0, 255, 0), 2)

        if contours:
            track_polygon = Polygon([pt[0] for pt in contours[0]])
        else:
            track_polygon = Polygon()
        
        # track_polygon = extend_track_region(track_polygon, extension_length)
        danger_levels = calculate_danger_level(pred, track_polygon, low_risk_th, high_risk_th)

        for result in danger_levels:
            obj = result["object"]
            level = result["danger_level"]
            print(f"Object {obj['class']} at {obj['bbox']} has danger level: {level}")
        
        plt.figure()
        plt.subplot(111)
        plt.imshow(im_rgb)
        plt.savefig('tmp/t.png')
        plt.show()
        
        # # Process predictions
        # for i, det in enumerate(pred):  # per image
        #     seen += 1
        #     if webcam:  # batch_size >= 1
        #         p, im0, frame = path[i], im0s[i].copy(), dataset.count
        #         s += f"{i}: "
        #     else:
        #         p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

        #     p = Path(p)  # to Path
        #     save_path = str(save_dir / p.name)  # im.jpg
        #     txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
        #     s += "%gx%g " % im.shape[2:]  # print string
        #     gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        #     imc = im0.copy() if save_crop else im0  # for save_crop
        #     annotator = Annotator(im0, line_width=line_thickness, example=str(names))

        #     # Mask plotting  ----------------------------------------------------------------------------
        #     color_mask = [[0,0,255],[0,255,0]]
        #     pred_mask = F.interpolate(pred_mask[None], size = im0.shape[:2], mode = 'nearest')
        #     pred_mask = torch.squeeze(pred_mask) # 1xcxhxw ->  cxhxw
        #     seg_nc, h, w = pred_mask.shape
        #     pred_mask = torch.argmax(pred_mask, dim=0) # torch.Size([640, 640]) 返回指定维度最大值的序号, (c,h,w) -> (h,w)
        #     image_mask = pred_mask.clone().detach().cpu().numpy() # 0 1 2 # (1200, 5760)

        #     with contextlib.suppress(Exception):
        #         im0[image_mask==1] = im0[image_mask==1] * 0.4 + np.array(color_mask[0]) * 0.6  # lane
        #         im0[image_mask==2] = im0[image_mask==2] * 0.4 + np.array(color_mask[1]) * 0.6  # drivable
        #     im0 = np.ascontiguousarray(im0)

        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        #         # Print results
        #         for c in det[:, 5].unique():
        #             n = (det[:, 5] == c).sum()  # detections per class
        #             s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        #         # Write results
        #         for *xyxy, conf, cls in reversed(det):
        #             c = int(cls)  # integer class
        #             label = names[c] if hide_conf else f"{names[c]}"
        #             confidence = float(conf)
        #             confidence_str = f"{confidence:.2f}"
        #             if save_img or save_crop or view_img:  # Add bbox to image
        #                 c = int(cls)  # integer class
        #                 label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
        #                 annotator.box_label(xyxy, label, color=colors(c, True))
        #             if save_crop:
        #                 save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)

        #     # Stream results
        #     im0 = annotator.result()
        
        # plt.figure()
        # plt.subplot(111)
        # plt.imshow(im0)
        # plt.savefig('tmp/t.png')
        # plt.show()

# TODO 找包围框，有问题，直接用Polygon可能出现的问题，解释出来，因为 障碍物所在区域，肯定Polygon不包含.. 
# TODO morphologyEx  kernel 设置的大一点 
# TODO 绘制出， bbox， segmentation mask， 写出分类等级 ... 
# DDL 6.1 之前完成



def parse_opt():
    """Parses command-line options for YOLOv5 inference including model paths, data sources, inference settings, and
    output preferences.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="runs/train-seg/obstacle/exp/weights/best.pt", help="model path(s)")
    parser.add_argument("--source", type=str, default= "/media/ubuntu/zoro/ubuntu/data/railway_obstacle_detection/ObstacleDetection/images/val/a0.jpg", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default="data/obstacle.yaml", help="dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/predict-seg", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--retina-masks", action="store_true", help="whether to plot masks in native resolution")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking for requirements before launching."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)