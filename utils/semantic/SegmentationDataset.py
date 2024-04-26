###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
# 带标准数据加载增广的语义分割Dataset, Dataset类代码原作者张航, 详见其开发的github仓库PyTorch-Encoding, 在此基础上魔改了一些包括不均匀的长边采样,色彩变换,pad0改成了pad255(配合bdd的格式)
# 稍加修改即可加载BDD100k分割数据, 此处写了Cityscapes+BDD100k混合训练，没加单独的BDD100k
###########################################################################

import os
import sys
from pathlib import Path

ROOT = "/media/ubuntu/zoro/ubuntu/code/obstacle_detection"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from tqdm import tqdm, trange
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.utils.data as data
from torchvision import transforms
from utils.general import make_divisible
from scipy import stats
import math
from functools import lru_cache
import matplotlib.pyplot as plt
from random import choices


@lru_cache(128)  # 目前每次调用参数都是一样的, 用cache加速, 有random的地方不能用cache
def range_and_prob(base_size, low: float = 0.5,  high: float = 3.0, std: int = 25) -> list:
    low = math.ceil((base_size * low) / 32)
    high = math.ceil((base_size * high) / 32)
    mean = math.ceil(base_size / 32) - 4  # 峰值略偏
    x = np.array(list(range(low, high + 1)))
    p = stats.norm.pdf(x, mean, std)
    p = p / p.sum()  # 概率密度　choices权重不用归一化, 归一化用于debug和可视化调参std,以及用cum_weights优化
    cum_p = np.cumsum(p)  # 概率分布，累加
    # print("!!!!!!!!!!!!!!!!!!!!!!")
    return (x, cum_p)


# 用均值为basesize的正态分布模拟一个类似F分布图形的采样, 目的是专注于目标scale的同时见过少量大scale(通过apollo图天空同时不掉点)
def get_long_size(base_size:int, low: float = 0.5,  high: float = 3.0, std: int = 40) -> int:  
    x, cum_p = range_and_prob(base_size, low, high, std)
    # plt.plot(x, p)
    # plt.show()
    longsize = choices(population=x, cum_weights=cum_p, k=1)[0] * 32  # 用cum_weights O(logn)，　用weights O(n)
    # print(longsize)
    return longsize


# 基础语义分割类
class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None,
                 target_transform=None, base_size=520, crop_size=480, low=0.6, high=3.0, sample_std=25):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.low = low
        self.high = high
        self.sample_std = sample_std
        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'. \
                format(base_size, crop_size))
            print(f"Random scale low: {self.low}, high: {self.high}, sample_std: {self.sample_std}")

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def make_pred(self, x):
        return x + self.pred_offset

    def _testval_img_transform(self, img):  
        # 新的训练后测验证集数据处理(仅支持同尺寸图): 图长边resize到base_size, 但标签是原图, 若非原图需要测试时手动把输出放大到原图 (原版仅处理标签, 原图输入)
        w, h = img.size
        outlong = self.base_size
        outlong = make_divisible(outlong, 32)  # 32是网络最大下采样倍数, 测试时自动使边为32倍数
        if w > h:
            ow = outlong
            oh = int(1.0 * h * ow / w)
            oh = make_divisible(oh, 32)
        else:
            oh = outlong
            ow = int(1.0 * w * oh / h)
            ow = make_divisible(ow, 32)
        img = img.resize((ow, oh), Image.BILINEAR)
        return img

    def _val_sync_transform(self, img, mask):  
        w_crop_size, h_crop_size = self.crop_size
        w, h = img.size
        long_size = get_long_size(base_size=self.base_size, low=self.low, high=self.high, std=self.sample_std)  # random.randint(int(self.base_size*0.5), int(self.base_size*2))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # center crop
        w, h = img.size
        x1 = int(round((w - w_crop_size) / 2.))
        y1 = int(round((h - h_crop_size) / 2.))
        img = img.crop((x1, y1, x1+w_crop_size, y1+h_crop_size))
        mask = mask.crop((x1, y1, x1+w_crop_size, y1+h_crop_size))
        return img, mask  # 这里改了, 在__getitem__里再调用self._mask_transform(mask)

    def _sync_transform(self, img, mask):  # 训练数据增广
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        w_crop_size, h_crop_size = self.crop_size
        # random scale (short edge)  从base_size一半到两倍间随机取数, 图resize长边为此数, 短边保持比例
        w, h = img.size
        long_size = get_long_size(base_size=self.base_size, low=self.low, high=self.high, std=self.sample_std)  
        # random.randint(int(self.base_size*0.5), int(self.base_size*2))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop  边长比crop_size小就pad
        if ow < w_crop_size or oh < h_crop_size:  # crop_size:
            padh = h_crop_size - oh if oh < h_crop_size else 0
            padw = w_crop_size - ow if ow < w_crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255) 
            # mask不填充0而是填255:类别0不是训练类别,后续会被填-1(但bdd100k数据格式是trainid,为了兼容填255)
        
        # random crop 
        w, h = img.size
        x1 = random.randint(0, w - w_crop_size)
        y1 = random.randint(0, h - h_crop_size)
        img = img.crop((x1, y1, x1+w_crop_size, y1+h_crop_size))
        mask = mask.crop((x1, y1, x1+w_crop_size, y1+h_crop_size))

        # return img, self._mask_transform(mask)
        return img, mask  # 这里改了, 在__getitem__里再调用self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()


class CustomSegmentation(BaseDataset):  
    def __init__(self, root, split='train', mode=None, transform=None, target_transform=None, **kwargs):
        super(CustomSegmentation, self).__init__(root, split, mode, transform, target_transform, **kwargs)
        
        self.images, self.mask_paths = get_custom_pairs(self.root, self.split, seg_fold = "sematic_labels")
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n")

    def __getitem__(self, index):
        imagepath = self.images[index]
        img = Image.open(imagepath).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)  # 训练数据增广
            mask = torch.from_numpy(np.array(mask)).long()
            mask[mask==255] = -1
        elif self.mode == 'val':
            # print(mask)
            img, mask = self._val_sync_transform(img, mask)  # 验证数据处理
            
            mask = torch.from_numpy(np.array(mask)).long()
            mask[mask==255] = -1
        else:
            assert self.mode == 'testval'   # 训练时候验证用val(快, 省显存),测试验证集指标时用testval一般mIoU会更高且更接近真实水平
            # mask = self._mask_transform(mask)  # 测试验证指标, 除转换标签格式外不做任何处理
            img = self._testval_img_transform(img)
            mask = torch.from_numpy(np.array(mask)).long()
            mask[mask==255] = -1

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.images)




def get_custom_pairs(folder, split='train', seg_fold = "sematic_labels"):
    
    def get_path_pairs(img_folder, mask_folder, seg_fold):
        img_paths = []
        mask_paths = []
        
        for root, dirs, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    imgpath = os.path.join(root, filename)
                    maskname = filename.replace('images', seg_fold)
                    if filename.endswith(".jpg"): 
                        maskname =maskname.replace('.jpg', '.png')
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else: 
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split == 'train' or split == 'val' or split == 'test':
        img_folder = os.path.join(folder, 'images/' + split)
        mask_folder = os.path.join(folder, seg_fold,  split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder, seg_fold)
        return img_paths, mask_paths
   

# 默认custom_loader　jitter和crop采用更保守的方案
def get_custom_loader(root, split="train", mode="train",  # 获取训练和验证用的dataloader
                     base_size=1024, batch_size=32, workers=4, pin=True):
    if mode == "train":
        input_transform = transforms.Compose([transforms.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.4, hue=0),
                                              transforms.ToTensor()])
    else:
        input_transform = transforms.Compose([transforms.ToTensor()])
    
    dataset = CustomSegmentation(root=root, split=split, mode=mode,transform=input_transform,
                               base_size=base_size, crop_size=(base_size, base_size), low=0.75, high=1.5, sample_std=35)
    
    loader = data.DataLoader(dataset, batch_size=batch_size,
                             drop_last=True if mode == "train" else False, shuffle=True if mode == "train" else False,
                             num_workers=workers, pin_memory=pin)
    return loader


if __name__ == "__main__":


    data_path = '/media/ubuntu/zoro/ubuntu/data/ObstacleDetection'
    # t = transforms.Compose([  # 用于打断点时候测试色彩和大小裁剪变换是否合理
    #     transforms.ColorJitter(brightness=0.45, contrast=0.45,
    #                            saturation=0.45, hue=0.1)])
    trainloader = get_custom_loader(root=data_path, split="val", mode="val", base_size=640, workers=0, pin=True, batch_size=4)

    import time
    t1 = time.time()
    # for i, data in enumerate(trainloader):
        # print(f"batch: {i}")

    # print(f"cost {(time.time()-t1)/(i+1)} per batch load")
 