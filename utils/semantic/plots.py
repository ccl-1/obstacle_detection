import contextlib
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .. import threaded
from ..general import xywh2xyxy
from ..plots import Annotator, colors
from torchvision.utils import draw_segmentation_masks, save_image

def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x

@threaded
def plot_images_and_masks(images, targets, semasks, nc_seg, paths=None, fname="images.jpg", names=None):
    """Plots a grid of images, their labels, and masks with optional resizing and annotations, saving to fname."""
    # 将一个batch的图片都放在一个大图mosaic上面
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(semasks, torch.Tensor):
        semasks = semasks.cpu().numpy().astype(int)

    max_size = 1920  # max image size
    max_subplots = 16  # max image subplots, i.e. 4x4
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs**0.5)  # number of subplots (square)
    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)

    # Build Image
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, im in enumerate(images): # 遍历当前 batch 中的所有图片
        if i == max_subplots:  # if last batch has fewer images than we expect
            break
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
        im = im.transpose(1, 2, 0)
        mosaic[y : y + h, x : x + w, :] = im

    # Resize (optional)
    scale = max_size / ns / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)
        mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))

    # Annotate
    fs = int((h + w) * ns * 0.01)  # font size
    annotator = Annotator(mosaic, line_width=round(fs / 10), font_size=fs, pil=True, example=names)
    # 遍历mosaic中的每一个图片，绘制bbox，label，mask
    for i in range(i + 1):
        x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin = 左上角位置 
        annotator.rectangle([x, y, x + w, y + h], None, (255, 255, 255), width=2)  # borders
        if paths:
            annotator.text([x + 5, y + 5], text=Path(paths[i]).name[:40], txt_color=(220, 220, 220))  # filenames
            
        # --------------- plot semantic mask ----------------------
        if len(semasks):
            image_masks = semasks[i]
            im = np.asarray(annotator.im).copy()
            color = [[255,255,255], [0,255,0], [0,0,255]]

            mh, mw = image_masks.shape 
            if mh != h or mw != w:
                mask = image_masks.astype(np.uint8)
                mask = cv2.resize(mask, (w, h))

            print(torch.unique(torch.tensor(mask)), "----------------")
            for idx in range(1, nc_seg):
                colored_mask = np.zeros((h,w, 3))
                colored_mask[mask== idx] = color[idx-1]
            
            mask = mask >0
            with contextlib.suppress(Exception):
                # im[y:y + h, x:x + w, :][mask] = im[y:y + h, x:x + w, :][mask] * 0.4 + np.array(colors) * 0.6
                im[y:y + h, x:x + w, :][mask] = im[y:y + h, x:x + w, :][mask] * 0.4 + colored_mask[mask] * 0.6

            annotator.fromarray(im) # Update self.im from a numpy array

        #  -------------- plot bbox and labels ------------------------
        if len(targets) > 0: 
            idx = targets[:, 0] == i
            ti = targets[idx]  # image targets

            boxes = xywh2xyxy(ti[:, 2:6]).T
            classes = ti[:, 1].astype("int")
            labels = ti.shape[1] == 6  # labels if no conf column
            conf = None if labels else ti[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale < 1:  # absolute coords need scale if image scales
                    boxes *= scale
            boxes[[0, 2]] += x
            boxes[[1, 3]] += y
            # 将当前的图片img的所有标签boxes画到mosaic上
            for j, box in enumerate(boxes.T.tolist()):
                cls = classes[j]
                color = colors(cls)
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = f"{cls}" if labels else f"{cls} {conf[j]:.1f}"
                    annotator.box_label(box, label, color=color)
        
        
    annotator.im.save(fname)  # save


def plot_results_with_masks(file="path/to/results.csv", dir="", best=True):
    """
    Plots training results from CSV files, plotting best or last result highlights based on `best` parameter.

    Example: from utils.plots import *; plot_results('path/to/results.csv')
    """
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            index = np.argmax(
                0.9 * data.values[:, 8] + 0.1 * data.values[:, 7] + 0.9 * data.values[:, 12] + 0.1 * data.values[:, 11]
            )
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]):
                y = data.values[:, j]
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker=".", label=f.stem, linewidth=2, markersize=2)
                if best:
                    # best
                    ax[i].scatter(index, y[index], color="r", label=f"best:{index}", marker="*", linewidth=3)
                    ax[i].set_title(s[j] + f"\n{round(y[index], 5)}")
                else:
                    # last
                    ax[i].scatter(x[-1], y[-1], color="r", label="last", marker="*", linewidth=3)
                    ax[i].set_title(s[j] + f"\n{round(y[-1], 5)}")
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print(f"Warning: Plotting error for {f}: {e}")
    ax[1].legend()
    fig.savefig(save_dir / "results.png", dpi=200)
    plt.close()