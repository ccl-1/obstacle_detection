import contextlib
import argparse
import os
import sys
import cv2
from shapely.geometry import Polygon, box
from shapely.affinity import translate
from shapely.ops import nearest_points

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
from utils.general import ( Profile, check_file, check_img_size,
check_imshow, check_requirements, cv2, increment_path, scale_boxes,
non_max_suppression, print_args,
)
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score)


risk_level_dict = {0:'Safe', 1:' Low', 2:' High'}
def relationship_between_polygon_and_box(polygon, detections):
    poly = Polygon(polygon)
    for dets in detections:
        for j in range(len(dets)):
            x1, y1, x2, y2 = dets[j, :4]
            rect = box(x1, y1, x2, y2)
            if poly.intersects(rect):
                danger_level = 2
                return danger_level
            else: 
                nearest_points_poly, nearest_points_rect = nearest_points(poly, rect)
                distance = nearest_points_poly.distance(nearest_points_rect)
                if distance < 50:
                    danger_level = 2
                else:
                    danger_level = 1
    return danger_level


def get_classification_acc(preds, gts, plot_cm=False):  
    if len(preds) != len(gts):  
        raise ValueError("The predictions and truths lists must have the same length.")  
    for pred, gt in zip(preds, gts):  
        if pred[0] != gt[0]:  
            raise ValueError("The predictions and gt lists must have the same order.") 

    classes = ['0', '1', '2'] 
    # classes = ["safe","Low", "High"]

    y_pred = [i[1] for i in preds]
    y_true =   [i[1] for i in gts]
    
    acc = accuracy_score(y_pred, y_true) # (TP+TN)/total
    f1 = f1_score( y_true, y_pred, average='macro' )
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    
    confusion = confusion_matrix(y_pred, y_true)
    if plot_cm:
        plt.imshow(confusion, cmap=plt.cm.Greens)
        indices = range(len(confusion))
        plt.xticks(indices, classes)
        plt.yticks(indices, classes)
        plt.colorbar()
        plt.xlabel('y_pred')
        plt.ylabel('y_true')
        for first_index in range(len(confusion)):
            for second_index in range(len(confusion[first_index])):
                plt.text(first_index, second_index, confusion[first_index][second_index])
        plt.savefig('tmp/cm.png')
        plt.show()
    return acc, f1, p, r



@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s-seg.pt",  # model.pt path(s)
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    gt_path=None,
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

    class_results = []
    # Run inference
    model.eval()
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        a = 0.3
        file_name = path.split('/')[-1][:-4]
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
        
        
        # im0 = torch.squeeze(im, dim=0).permute(1,2,0).cpu().numpy().copy()*255
        # with contextlib.suppress(Exception):
        #         im0[segmentation_result==1] = im0[segmentation_result==1] + np.array([0,0,255]) * a
        # im0 = np.ascontiguousarray(im0)
        # cv2.imwrite(os.path.join('runs/tmp/', file_name+'.png'), im0)
        
        # Morphological Operations
        kernel = np.ones((10, 10), np.uint8)
        closed_mask = cv2.morphologyEx(segmentation_result, cv2.MORPH_CLOSE, kernel)
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel) # remove noise
        contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        im_vis = torch.squeeze(im, dim=0).permute(1,2,0).cpu().numpy().copy() * 255

        risk_level = 0    
        contours_filtered = []    
        if len(contours) > 0:
            for cnt in contours:
                area = cv2.contourArea(cnt)  
                print(area)
                if area > 2000:
                    contours_filtered.append(cnt)

        if len(contours) > 0 and len(pred) > 0 and pred[0].shape[0] > 0:
            for contour in contours_filtered:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(im_vis, [approx], 0, (0, 0, 255), 3)
                zeros = np.zeros((im_vis.shape), dtype=np.uint8)
                mask = cv2.fillPoly(zeros, [approx], color=(0, 0, 255))
                im_vis = a * mask + im_vis
                
                risk_level_object = relationship_between_polygon_and_box(approx.reshape(-1, 2), pred)
                if risk_level_object == 2:
                    risk_level = risk_level_object
                    break
                else:
                    risk_level = max(risk_level, risk_level_object)
            # cv2.imwrite(os.path.join('runs/tmp/', file_name+'_1.png'), im_vis)

        else:
            risk_level = 0
    
        for i, dets in enumerate(pred):  # per image
            if len(dets):
                for j in range(len(dets)):
                    c = int(dets[j, 5])  # integer class
                    conf = float(dets[j, 4])
                    label = f"{names[c]} {conf:.2f}"
                    x1, y1, x2, y2 = dets[j, :4]
                    cv2.rectangle(im_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(im_vis, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        # print("Risk Level: ", risk_level_dict[risk_level])
        class_results.append([file_name, risk_level])
        class_results = sorted(class_results, key=lambda x: x[0]) 

        if False:
            if risk_level < 2: # only save safe and low result
                c_vis= [(0,255, 0), (255, 0, 0), (0, 0, 255)]
                info = "Risk Level: " + risk_level_dict[risk_level]
                font, font_scale, font_thickness, xx, yy = cv2.FONT_HERSHEY_SIMPLEX, 1, 2, 20, 40
                (text_w, text_h), _ = cv2.getTextSize(info, font, font_scale, font_thickness)
                cv2.rectangle(im_vis, (xx-5, yy-text_h-5), (xx+text_w, yy+15), c_vis[risk_level], -1)
                cv2.putText(im_vis, info, (xx, yy), font, font_scale, (255, 255, 255), font_thickness)
                cv2.imwrite(os.path.join('runs/predict-seg/Post/', file_name+'.png'), im_vis)

    """
    # save classification result to txt
    # result_save_path = "tmp/pred_val.txt"
    result_save_path = "tmp/pred_train.txt"
    with open(result_save_path, 'w') as f:
        for item in class_results:  
            line = ' '.join(map(str, item))  
            f.write(line + '\n')
    
    # load gt
    class_gt = []
    # GT_path = "tmp/val.txt"
    GT_path = "tmp/train.txt"
    with open(GT_path, 'r') as file:  
        for line in file:  
            info = line.strip().split(' ')
            fn, idx = info[0], int(info[1])
            class_gt.append([fn, idx])
    class_gt = sorted(class_gt, key=lambda x: x[0]) 

    acc, f1, p, r = get_classification_acc(class_results, class_gt, plot_cm=False)
    print("Acc:{:.2f},  F1-score:{:.2f},  P:{:.2f},  R:{:.2f}".format(acc, f1, p, r))  
    """

def parse_opt():
    """Parses command-line options for YOLOv5 inference including model paths, data sources, inference settings, and
    output preferences.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="runs/train-seg/transfer/RS19+OBJECT-2-ROD/weights/best.pt", help="model path(s)")
    # parser.add_argument("--weights", nargs="+", type=str, default="runs/train-seg/obstacle/exp/weights/best.pt", help="model path(s)")

#     parser.add_argument("--source", type=str, default= "/media/ubuntu/zoro/ubuntu/data/\
# railway_obstacle_detection/ObstacleDetection/images/train/",  help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--source", type=str, default= "/media/ubuntu/zoro/ubuntu/data/railway_obstacle_detection/ObstacleDetection/images/train/c34.jpg",  help="file/dir/URL/glob/screen/0(webcam)")
    # parser.add_argument("--source", type=str, default= "/media/ubuntu/zoro/ubuntu/data/railway_obstacle_detection/ObstacleDetection/images/train/c53.jpg",  help="file/dir/URL/glob/screen/0(webcam)")
    # parser.add_argument("--source", type=str, default= "/media/ubuntu/zoro/ubuntu/data/railway_obstacle_detection/ObstacleDetection/images/train/c129.jpg",  help="file/dir/URL/glob/screen/0(webcam)")

    parser.add_argument("--data", type=str, default="data/obstacle.yaml", help="dataset.yaml path")
    parser.add_argument("--gt_path", type=str, default="tmp/val.txt", help="dataset.yaml path")
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