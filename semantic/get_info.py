# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
python semantic/get_info.py      --weights runs/train-seg/head/bdd100k/S2/weights/last.pt  --data ./data/bdd100k-seg.yaml     

"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import  check_img_size,check_requirements,check_yaml,print_args
from utils.torch_utils import select_device, smart_inference_mode
from thop import profile
# from torchstat import stat


@smart_inference_mode()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    task="val",
):
    device = select_device(device, batch_size=batch_size)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    half = model.fp16  # FP16 supported on limited backends with CUDA
    device = model.device
    model.eval()

# ------------ parameters ------------------- 
    input = torch.randn(1, 3, 640, 640).cuda()
    flops, params = profile(model, inputs=(input, ), verbose=False)
    print("FLOPs=", '{:.2f}G, '.format(flops/1e9*2), "params=", '{:.2f}M'.format(params/1e6))

# ------------ test fps -------------------
    steps = 50
    torch.cuda.synchronize()
    time_start = time.time()
    for i in range(steps):
        output = model(input) 
    torch.cuda.synchronize()
    time_end = time.time()
    time_sum = (time_end - time_start) / steps
    print('Inference Speed: {:.2f} ms, '.format(time_sum * 1000), '{:.2f} FPS'.format(1.0/time_sum))



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/bdd100k-seg.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default="runs/train-seg/bdd100k_det_first/exp/weights/last.pt", help="model path(s)")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")         
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--task", default="val", help="train, val, test, speed or study")

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 tasks including training, validation, testing, speed, and study with configurable options."""
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    if opt.task in ("train", "val", "test"):  # run normally
        run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
