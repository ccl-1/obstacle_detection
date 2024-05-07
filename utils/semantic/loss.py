import torch
import torch.nn as nn
import torch.nn.functional as F
from distutils.version import LooseVersion
import numpy as np


from ..loss import FocalLoss, smooth_BCE
from ..metrics import bbox_iou
from ..torch_utils import de_parallel


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, overlap=False):
        """Initializes the compute loss function for YOLOv5 models with options for autobalancing and overlap
        handling.
        """
        self.sort_obj_iou = False
        self.overlap = overlap
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device)) #  one single class.
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))
        CEseg = nn.CrossEntropyLoss() # ignore_index=0
        BCEseg = nn.BCEWithLogitsLoss()
        self.BCEseg = BCEseg
        self.CEseg = CEseg


        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        # self.nm = m.nm  # number of masks
        self.anchors = m.anchors
        self.device = device

    def __call__(self, preds, targets, masks):  # predictions, targets, model
        """Evaluates YOLOv5 model's loss for given predictions, targets, and masks; returns total loss components."""
        pred_detects, pred_masks = preds
        # pred_masks.shape: (batch_size, num_classes, img_h, img_w))
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)
        l_seg = torch.zeros(1, device=self.device)
        l_dice = torch.zeros(1, device=self.device)

        # print("---------------------")
        # print(len(pred_detects)) # 3
        # print(pred_detects[0].size()) # torch.Size([32, 3, 56, 84, 85])
        # print(pred_detects[1].size()) # torch.Size([32, 3, 42, 42, 85])
        # print(pred_detects[2].size()) # torch.Size([32, 3, 21, 19, 85])

        tcls, tbox, indices, anchors = self.build_targets(pred_detects, targets)  # targets


        # 遍历 Detection输出层， 计算 loss
        for i, pi in enumerate(pred_detects):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            #  Detection loss -------------------------------------
            n = b.shape[0]  # number of targets
            if n: # 遍历所有目标 计算 loss
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # subset of predictions

                # Box regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE ,  torch.float16, 
                    # shape=([153, 26]), [153, 26]) ([393, 26]) ([393, 26]) ([261, 26]) ([261, 26])
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        # Semantic Segmentation Loss ------------------------------------------
        bs, nc_seg, mask_h, mask_w = pred_masks.shape  # batch size, number of masks, mask height, mask width
        pred_masks = pred_masks.type(masks.dtype) #  torch.float32

        # print(torch.unique(masks[1,:,:])) # check mask labels after augmentation

        # pixel classify loss
        pixel_loss = self.CEseg(pred_masks, masks.long()) # without onehot, 多分类。  [bs, C，*]  [bs，*] 
        l_seg += pixel_loss
        
        # dice loss 
        # print(torch.unique(masks), "----------------")
        masks = torch.nn.functional.one_hot(masks.to(torch.int64), nc_seg).permute(0, 3, 1, 2).float()   
        pt = torch.flatten(pred_masks.softmax(dim = 1))
        gt = torch.flatten(masks)

        inter_mask = torch.sum(torch.mul(pt, gt))
        union_mask = torch.sum(torch.add(pt, gt))
        dice_coef = (2. * inter_mask + 1.) / (union_mask + 1.)
        l_dice += (1. - dice_coef) / 2.

        lbox *= self.hyp["box"] # box: 0.05 
        lobj *= self.hyp["obj"] # obj: 1.0 
        lcls *= self.hyp["cls"] # cls: 0.5 
        l_seg *= self.hyp["box"]  
        l_dice *= self.hyp["box"] 

        loss = lbox + lobj + lcls + l_seg + l_dice
        # print("------------------", lbox.detach())
        return loss * bs, torch.cat((lbox, lobj, lcls, l_seg, l_dice)).detach()
    
    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
