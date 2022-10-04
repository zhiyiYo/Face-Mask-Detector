# coding: utf-8
from typing import List
import numpy as np

import torch
from torch import Tensor, nn
from utils.box_utils import match, decode, ciou, iou


class YoloLoss(nn.Module):
    """ 损失函数 """

    def __init__(self, anchors: list, n_classes: int, image_size: int, overlap_thresh=0.5):
        """
        Parameters
        ----------
        anchors: list of shape `(3, n_anchors, 2)`
            先验框列表，尺寸从大到小

        n_classes: int
            类别数

        image_size: int
            输入神经网络的图片大小

        overlap_thresh: float
            视为忽视样例的 IOU 阈值
        """
        super().__init__()
        self.n_anchors = len(anchors[0])
        self.anchors = np.array(anchors).reshape(-1, 2)
        self.n_classes = n_classes
        self.image_size = image_size
        self.overlap_thresh = overlap_thresh

        # 损失函数各个部分的权重
        self.balances = [0.4, 1, 4]
        self.lambda_box = 0.05
        self.lambda_obj = 5*(image_size/416)**2
        self.lambda_cls = n_classes / 80

        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, index: int,  pred: Tensor, targets: List[Tensor]):
        """ 计算一个特征图的损失

        Parameters
        ----------
        index: int
            特征图的索引，取值范围为 0~2
        preds: Tensor
            Yolo 神经网络输出的各个特征图，维度为:
            * `(N, (n_classes+5)*n_anchors, 13, 13)`
            * `(N, (n_classes+5)*n_anchors, 26, 26)`
            * `(N, (n_classes+5)*n_anchors, 52, 52)`

        targets: List[Tensor]
            标签数据，每个标签张量的维度为 `(N, n_objects, 5)`，最后一维为边界框 `(cx, cy, w, h, class)`

        Returns
        -------
        loss: Tensor of shape `(1, )`
            损失值
        """
        loss = 0
        N, _, h, w = pred.shape

        # 对预测结果进行解码，shape: (N, n_anchors, H, W, n_classes+5)
        anchor_mask = list(
            range(index*self.n_anchors, (index+1)*self.n_anchors))
        pred = decode(pred, self.anchors[anchor_mask],
                      self.n_classes, self.image_size, False)

        # 匹配边界框
        step_h = self.image_size / h
        step_w = self.image_size / w
        anchors = [[i/step_w, j/step_h] for i, j in self.anchors]
        p_mask, n_mask, gt = match(
            anchors, anchor_mask, targets, h, w, self.n_classes, self.overlap_thresh)
        self.mark_ignore(pred, targets, n_mask)

        p_mask = p_mask.to(pred.device)
        n_mask = n_mask.to(pred.device)
        gt = gt.to(pred.device)

        m = p_mask == 1
        if m.sum() != 0:
            # 定位损失
            iou = ciou(pred[..., :4], gt[..., :4])
            m &= torch.logical_not(torch.isnan(iou))
            loss += torch.mean((1-iou)[m])*self.lambda_box

            # 分类损失
            loss += self.bce_loss(pred[..., 5:][m], gt[..., 5:][m])*self.lambda_cls

        # 正样本和负样本的置信度损失
        mask = n_mask.bool() | m
        loss += self.bce_loss(pred[..., 4]*mask, m.type_as(pred)*mask) * \
            self.lambda_obj*self.balances[index]

        return loss

    def mark_ignore(self, pred: Tensor, targets: List[Tensor], n_mask: Tensor):
        """ 标记出忽略样本

        Parameters
        ----------
        pred: Tensor of shape `(N, n_anchors, H, W, n_classes+5)`
            解码后的特征图

        targets: List[Tensor]
            标签数据，每个标签张量的维度为 `(N, n_objects, 5)`，最后一维为边界框 `(cx, cy, w, h, class)`

        n_mask: Tensor of shape `(N, n_anchors, H, W)`
            负样本遮罩
        """
        N, _, h, w, _ = pred.shape
        bbox = pred[..., :4]

        for i in range(N):
            if targets[i].size(0) == 0:
                continue

            # shape: (h*w*n_anchors, 4)
            box = bbox[i].view(-1, 4)
            target = torch.zeros_like(targets[i][..., :4])
            target[:, [0, 2]] = targets[i][:, [0, 2]] * w
            target[:, [1, 3]] = targets[i][:, [1, 3]] * h

            # 计算预测框和真实框的交并比
            max_iou, _ = torch.max(iou(target, box), dim=0)
            max_iou = max_iou.view(pred[i].shape[:3])
            n_mask[i][max_iou > self.overlap_thresh] = 0
