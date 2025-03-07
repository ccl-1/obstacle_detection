# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Dataloaders."""

import os
import random
import glob
import cv2
import numpy as np
import math

import hashlib
from itertools import repeat
from multiprocessing.pool import Pool
from PIL import Image


import sys
from pathlib import Path

ROOT = "/media/ubuntu/zoro/ubuntu/code/obstacle_detection/"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


import torch
from torch.utils.data import DataLoader, distributed

from utils.dataloaders import InfiniteDataLoader,LoadImagesAndLabels, SmartDistributedSampler, seed_worker, exif_size
from utils.general import LOGGER, TQDM_BAR_FORMAT, xywhn2xyxy,xyxy2xywhn, xyn2xy, bdd100k_10_to_5_class

from utils.torch_utils import torch_distributed_zero_first
from utils.semantic.augmentations import mask_random_perspective, random_perspective, letterbox, augment_hsv, Albumentations
# from utils.augmentations import random_perspective
from utils.metrics import bbox_ioa

RANK = int(os.getenv("RANK", -1))

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html



def create_dataloader(
    path,
    imgsz,
    batch_size,
    stride,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    shuffle=False,
    mask_downsample_ratio=1,
    overlap_mask=False,
    seed=0,
    label_mapping=None,
    use_bdd100k_5=False,
    data_mode=None
):
    if rect and shuffle:
        LOGGER.warning("WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        
        if data_mode == None:
            dataset = LoadImagesAndLabelsAndMasks(
                path,
                imgsz,
                batch_size,
                augment=augment,  # augmentation
                hyp=hyp,  # hyperparameters
                rect=rect,  # rectangular batches
                cache_images=cache,
                single_cls=single_cls,
                stride=int(stride),
                pad=pad,
                image_weights=image_weights,
                prefix=prefix,
                downsample_ratio=mask_downsample_ratio,
                overlap=overlap_mask,
                rank=rank,
                label_mapping=label_mapping,
                use_bdd100k_5 =use_bdd100k_5
            )
        else:
            dataset = OnlyLoadImages_Masks(
                path,
                imgsz,
                batch_size,
                augment=augment,  # augmentation
                hyp=hyp,  # hyperparameters
                rect=rect,  # rectangular batches
                cache_images=cache,
                single_cls=single_cls,
                stride=int(stride),
                pad=pad,
                image_weights=image_weights,
                prefix=prefix,
                downsample_ratio=mask_downsample_ratio,
                overlap=overlap_mask,
                rank=rank,
                label_mapping=label_mapping,
                use_bdd100k_5 =use_bdd100k_5,
                data_mode=data_mode
            )

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else SmartDistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    if data_mode == None:
        return loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=nw,
            sampler=sampler,
            pin_memory=True,
            collate_fn=LoadImagesAndLabelsAndMasks.collate_fn4 if quad else LoadImagesAndLabelsAndMasks.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
        ), dataset
    else:
        return loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=nw,
            sampler=sampler,
            pin_memory=True,
            collate_fn=OnlyLoadImages_Masks.collate_fn4 if quad else OnlyLoadImages_Masks.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
        ), dataset



def semantic_img2label_paths(img_paths): #semantic labels
    """Generates label file paths from corresponding image file paths by replacing `/images/` with `/labels/` and
    extension with `.txt`.
    """
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}sematic_labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".png" for x in img_paths]


def img2label_paths(img_paths): # detection
    """Generates label file paths from corresponding image file paths by replacing `/images/` with `/labels/` and
    extension with `.txt`.
    """
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt" for x in img_paths]

def get_hash(paths):
    """Generates a single SHA256 hash for a list of file or directory paths by combining their sizes and paths."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

def merge_bdd100k(lb):
    det_cls = bdd100k_10_to_5_class()
    for label_box in lb:   
        label_box[0] = det_cls[int(label_box[0])]
    return lb

class LoadImagesAndLabelsAndMasks(LoadImagesAndLabels):  # for training/testing
    cache_version = 0.6
    def __init__(self, path, img_size=640, batch_size=16, augment=False, 
                 hyp=None, rect=False, image_weights=False, cache_images=False, single_cls=False, 
                 stride=32, pad=0, min_items=0, prefix="", downsample_ratio=1, 
                 overlap=False, rank=-1, seed=0, label_mapping=None,use_bdd100k_5=False,
    ):
        super().__init__(path, img_size, batch_size, augment, hyp, rect, 
                         image_weights, cache_images, single_cls, stride, pad, min_items, 
                         prefix, rank, seed) # , label_mapping, use_bdd100k_5=use_bdd100k_5
        self.downsample_ratio = downsample_ratio
        self.overlap = overlap
        self.label_mapping = label_mapping # for segmentation not detection
        self.use_bdd100k_5 = use_bdd100k_5
        # self.class_weights = torch.FloatTensor([0.8373, 0.918]).cuda()
        
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None


        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')
        
        self.semantic_label_files = semantic_img2label_paths(self.im_files)  # labels

        #todo 遍历label， 存储bbox到labels 。。。
        self.detect_label_files = img2label_paths(self.im_files)  # labels

        # load labels
        self.labels = []
        for lb_file in self.detect_label_files:
            if os.path.isfile(lb_file):
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if use_bdd100k_5:     
                        lb = merge_bdd100k(lb) 
                    lb = np.array(lb, dtype=np.float32)
                self.labels.append(lb)

        n = len(self.im_files)
        # Filter images
        for idx in range(n-1,-1,-1): # 倒序删除
            if  os.path.exists(self.im_files[idx]) and os.path.exists(self.semantic_label_files[idx]):
                pass
            else:
                self.im_files.pop(idx)
                self.semantic_label_files.pop(idx)
                self.labels.pop(idx)  # 缺少得全黑表示 .. .

        self.indices = range(len(self.im_files))
  


    def __len__(self):
        return len(self.im_files)
    
    def __getitem__(self, index):
        """Returns a transformed item from the dataset at the specified index, handling indexing and image weighting."""
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        if mosaic:
            # Load mosaic
            img, labels, semantic_labels = self.load_mosaic(index) 
            shapes = None
        else:
            # semantic_labels.size()=torch.Size([640, 640]), 单通道，不是onehot编码的格式。（nc,w,h）
            img, semantic_labels, (h0, w0), (h, w) = self.load_mask_image(index) # hw_original, hw_resized -> hw_resized

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape # [320 640]
            img, semantic_labels, ratio, pad = letterbox(img, semantic_labels, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
          
            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

            if self.augment:
                img, labels, semantic_labels = random_perspective(img, labels, semantic_labels,
                            degrees=hyp['degrees'],
                            translate=hyp['translate'],
                            scale=hyp['scale'],
                            shear=hyp['shear'],
                            perspective=hyp['perspective'])
       
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1e-3)
        
        # TODO: albumentations support
        if self.augment:
            # there are some augmentation that won't change boxes and masks,
            # so just be it for now.
            img, labels = self.albumentations(img, labels)
            nl = len(labels)  # update after albumentations
            # HSV color-space
            augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])
            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                semantic_labels = np.flipud(semantic_labels)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                semantic_labels = np.fliplr(semantic_labels)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img) # 将一个内存不连续存储的数组转换为内存连续存储的数组 shape (3, 640, 640)
        semantic_labels = np.ascontiguousarray(semantic_labels) # shape: (640, 640))

        assert semantic_labels.shape[-2:] == img.shape[-2:],   f' {semantic_labels.shape}============ {img.shape}'
        return torch.from_numpy(img), labels_out, torch.from_numpy(semantic_labels), self.im_files[index], shapes


    def convert_label(self, label, inverse=False):
        # label is mask, set the original key in mask to value
        if self.label_mapping is None:
            return label
        else:
            temp = label.copy()
            if inverse:
                for v, k in self.label_mapping.items(): # return (key, value), Todo key -> value
                    label[temp == k] = v
            else:
                for k, v in self.label_mapping.items(): # key <- value
                    label[temp == k] = v
            return label
    
    def load_mosaic(self, index):
        # YOLO 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        labels4, semantic_masks4 = [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

        # 3 additional image indices
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, semantic_labels, _, (h, w) = self.load_mask_image(index)


            # place img in img4
            if i == 0:  # top left
                # 创建马赛克图像 [1472, 1472, 3]=[h, w, c]
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                semantic_masks4 = np.full((s * 2, s * 2), 0, dtype=np.uint8)  # base mask with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # 将截取的图像区域填充到马赛克图像的相应位置
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            semantic_masks4[y1a:y2a, x1a:x2a] = semantic_labels[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels4.append(labels)

        # Concat/clip labels
        # 防止越界  label[:, 1:]中的所有元素的值（位置信息）必须在[0, 2*s]之间,小于0就令其等于0,大于2*s就等于2*s   out: 返回
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, 1:]):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

        # 3 additional image indices
        # Augment
        img4, labels4, semantic_masks4 = random_perspective(img4,
                                                      labels4,
                                                      semantic_masks4,
                                                      degrees=self.hyp["degrees"],
                                                      translate=self.hyp["translate"],
                                                      scale=self.hyp["scale"],
                                                      shear=self.hyp["shear"],
                                                      perspective=self.hyp["perspective"],
                                                      border=self.mosaic_border)  # border to remove

        # print(torch.unique(torch.tensor(semantic_masks4))) # tensor([ 0, 1], dtype=torch.uint8)
        # print(img4.shape, labels4.shape, semantic_masks4.shape) # (640, 640, 3) (3, 5) (640, 640)

        return img4, labels4, semantic_masks4


    def load_mask_image(self, i):
        # loads 1 image from dataset index 'i', returns im, original hw, resized hw
        path = self.im_files[i]
        lpath = self.semantic_label_files[i]

        im = cv2.imread(path)  # BGR
        # semantic_labels = cv2.imread(lpath, cv2.IMREAD_GRAYSCALE)  # BGR
        semantic_labels = cv2.imread(lpath, 0)  # BGR

        assert im is not None, f'Image Not Found {path}'
        assert semantic_labels is not None, f'Mask labels Not Found {lpath}'

        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            semantic_labels = cv2.resize(semantic_labels, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_NEAREST) 
        semantic_labels = self.convert_label(semantic_labels) 
        return im, semantic_labels, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized


    @staticmethod
    def collate_fn(batch):
        img, label, semantic_masks, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), torch.stack(semantic_masks, 0), path, shapes, 


def verify_image_size(im_file):
    im = Image.open(im_file)
    im.verify()  # PIL verify
    shape = exif_size(im)  # image size
    assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
    return shape

class OnlyLoadImages_Masks(LoadImagesAndLabelsAndMasks):
    cache_version = 0.6
    def __init__(self, path, img_size=640, batch_size=16, augment=False, 
                 hyp=None, rect=False, image_weights=False, cache_images=False, single_cls=False, 
                 stride=32, pad=0, min_items=0, prefix="", downsample_ratio=1, overlap=False, rank=-1,  seed=0, 
                 label_mapping=None, use_bdd100k_5=False, data_mode='det',
    ):
        # 对继承自父类的属性进行初始化，并且用父类的初始化方法初始化继承的属性。
        # super().__init__(path, img_size, batch_size, augment, hyp, rect, 
        #                  image_weights, cache_images, single_cls, stride, pad, min_items, prefix, rank, seed)
        self.downsample_ratio = downsample_ratio
        self.overlap = overlap
        self.label_mapping = label_mapping 
        self.use_bdd100k_5 = use_bdd100k_5 
        
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None

        # load images to self.im_files
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # get labels path
        self.labels = []
        self.data_mode = data_mode
        if self.data_mode == "det":
            self.semantic_label_files = None
            self.detect_label_files = img2label_paths(self.im_files)  # detection labels
            # load detection labels
            for lb_file in self.detect_label_files:
                if os.path.isfile(lb_file):
                    with open(lb_file) as f:
                        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        lb = np.array(lb, dtype=np.float32)
                    self.labels.append(lb)
        elif self.data_mode == "seg":
            self.detect_label_files = None
            self.semantic_label_files = semantic_img2label_paths(self.im_files)  # semantic labels
            lb= [[0, 0.5, 0.5, 0.9, 0.9],] #  不能设设置为空，那就图像尺寸得背景 ... 
            lb = np.array(lb, dtype=np.float32)
            self.labels = [ lb for i in self.semantic_label_files]
        else:
            print("data_mode must be 'seg' or 'det' ....  ")

        n = len(self.im_files)  # number of images  
        self.shapes = []
        for im_file in self.im_files:
            h0, w0 = verify_image_size(im_file)
    
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1: 
                h0, w0 = int(w0 * r), int(h0 * r)
            self.shapes.append((h0, w0))
             

        self.shapes = np.array(self.shapes)
        # Create indices
        
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(len(self.im_files))

        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        

    def load_mask_image(self, i):
        path = self.im_files[i]
        
        im = cv2.imread(path)  # BGR
        assert im is not None, f'Image Not Found {path}'
        h0, w0 = im.shape[:2]  # orig hw

        if self.data_mode == "seg":
            lpath = self.semantic_label_files[i]
            semantic_labels = cv2.imread(lpath, 0)  # BGR
        else:
            semantic_labels = np.zeros((h0,w0), dtype=np.uint8)
            semantic_labels[:,-5: ] = 255 # 至少2类，否则 loss 计算报错 ... 

        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            semantic_labels = cv2.resize(semantic_labels, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_NEAREST) 
        semantic_labels = self.convert_label(semantic_labels) 
        return im, semantic_labels, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    


# 统计数据集类别数量
def get_class(dataset, class_name):
    cls_num = {}
    for imgs, labels_out, targets, paths, shape in dataset:
        for box_info in labels_out:
            cls = int(box_info[1].to('cpu').item())
            # print(cls)
            label = class_name[cls] 
            if label not in cls_num:
                cls_num[label] =1
            else:
                cls_num[label] +=1
    cls_num = sorted(cls_num.items(), key = lambda x:x[0])  
    return cls_num


def mask2onehot(mask, nc):
    """Converts a segmentation mask (H,W) to (NC,H,W)"""
    h, w = mask.size()
    one_hot = torch.zeros(( nc, h, w))
    for i in range(nc):
        one_hot[i][mask==i] = 1
    return one_hot

def onehot2mask(mask):
    """
    Converts a mask (B, NC, H, W) to (B,H,W)
    """
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask


if __name__ == "__main__":

    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import yaml

    class_names = ["Fallen Trees", "Stone", "broken tracks", "car", "cow", "deer", 
       "elephant", "flood", "goat", "human", "landslide", "lion", "monkey", "peacock", "tiger", "excavator"]
    # vis_labels = ["broken tracks", "buffalo","flood", "landslide", "lion", "peacock", "truck",] # d=断轨道
    vis_labels = ["truck",] # d=断轨道


    hyp = "/media/ubuntu/zoro/ubuntu/code/obstacle_detection/data/hyps/hyp.scratch-low.yaml"
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    
    # Dataset只负责数据的抽取，调用一次__getitem__()只返回一个样本。如果希望批处理，还要同时进行shuffle和并行加速等操作，就需要使用DataLoader。
    # data_path = '/media/ubuntu/zoro/ubuntu/data/railway_obstacle_detection/ObstacleDetection/images/train'
    # # data_path = '/media/ym-004/9AC2F49AC2F47BB7/zoro/data/ObstacleDetection/images/val'
    # label_mapping_bdd100k = {0: 0, 215: 1} 

    # data_path = '/media/ubuntu/zoro/ubuntu/data/railway_obstacle_detection/bdd100k/For_OD/images/train'
    # label_mapping_bdd100k = {0: 0, 255:1, 127:2} 

    data_path = '/media/ubuntu/zoro/ubuntu/data/railway_obstacle_detection/object-13/images/train'
    label_mapping_bdd100k = {0: 0, 255:1} 

    #  label_mapping 必须是连续得
    # train_loader, dataset = create_dataloader(data_path,640, 4, 32, False, hyp, augment=True, \
    #                         label_mapping=label_mapping_bdd100k, use_bdd100k_5=False, data_mode='det')  # mosica augmentation， for train
    
    train_loader, dataset = create_dataloader(data_path, 640, 4, 32, False, hyp, rect=True, rank=-1,
                                              label_mapping=label_mapping_bdd100k, use_bdd100k_5=False, data_mode='det') # for val
    
    # train_loader, dataset = create_dataloader(data_path, 640, 4, 32, False, hyp, rect=True, rank=-1,
    #                                           label_mapping=label_mapping_bdd100k, use_bdd100k_5=False) # for val
    cls_num = get_class(dataset, class_names)
    print(cls_num)
    input()

    # nb = len(train_loader)  # number of batches
    # plt.figure(figsize=(15,10))
    # for imgs, targets, paths, shape in dataset:
    for imgs, labels_out, targets, paths, shape in dataset:
        # print(imgs.size(), labels_out.size(), targets.size()) # torch.Size([3, 640, 640]) torch.Size([3, 6]) torch.Size([640, 640])
        
        img = imgs.to('cpu').float() / 255 
        img = np.transpose(img,(1,2,0))
        mask = targets.to('cpu').float() 
        labels_out = labels_out.to('cpu').float().numpy()

        # onehot = mask2onehot(mask, 3)
        # print(torch.unique(mask)) # 统计mask类别 ... 
        # x = torch.unique(mask)
        # if len(x) !=2:
        #     print(paths)
        # break
        # l = len(torch.unique(torch.tensor(semantic_labels)))
        # if l >3:
        #     print("bug bug ----------------------------", lpath) # 统计mask类别 ... 
        # 
        # """ 
# ----------------------------------------------------------------------------------------
        # img[:,:,:][onehot[0]==1] = 0
        # img[:,:,:][onehot[1]==1] = 1
        img = np.ascontiguousarray(img)  # img为你的图片数组
        h, w, _ = img.shape
        flag= True
        # plt.figure(figsize=(10,10))

        for bbox in labels_out: # 遍历每一个obj
            label = class_names[int(bbox[1])]
            if label not in vis_labels:
                flag= False
                continue
            print(paths)
            # print("----------------", label, "--------------------------")
            # continue
        #     flag = True
        #     bbox2 = xywhn2xyxy(bbox[2:], w, h)  # normalized xywh to pixel xyxy format
        #     p1 = (int(bbox2[0]),int(bbox2[1]))
        #     p2 = (int(bbox2[2]),int(bbox2[3]))
        #     cv2.rectangle(img, p1, p2, [0,0,255], 2)

        #     wf,hf = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        #     cv2.rectangle(img, (p1[0], p1[1]-hf-2), (p1[0]+wf, p1[1]), [0,0,255],-1)
        #     cv2.putText(img, label, (p1[0], p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # if flag:
        #     plt.imshow(img)
        #     plt.show()
        


# """