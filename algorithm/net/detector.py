# coding:utf-8
from typing import Tuple, Dict, List
import numpy as np

import torch
from utils.box_utils import decode, center_to_corner
from torchvision.ops import nms


class Detector:
    """ 探测器 """

    def __init__(self, anchors: list, image_size: int, n_classes: int, conf_thresh=0.25, nms_thresh=0.45):
        """
        Parameters
        ----------
        anchors: list of shape `(3, n_anchors, 2)`
            先验框

        image_size: int
            图片尺寸

        n_classes: int
            类别数

        conf_thresh: float
            置信度阈值

        nms_thresh: float
            nms 操作中 iou 的阈值，越大保留的预测框越多
        """
        self.anchors = anchors
        self.n_classes = n_classes
        self.image_size = image_size
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh

    def __call__(self, preds: Tuple[torch.Tensor]) -> List[Dict[int, torch.Tensor]]:
        """ 对神经网络输出的结果进行处理

        Parameters
        ----------
        preds: Tuple[Tensor]
            神经网络输出的三个特征图, 维度为 `(N, C, 13, 13)`, `(N, C, 26, 26)` 和 `(N, C, 52, 52)`

        Returns
        -------
        out: List[Dict[int, Tensor]]
            所有输入图片的检测结果，列表中的一个元素代表一张图的检测结果，
            字典中的键为类别索引，值为该类别的检测结果，检测结果的最后一维的第一个元素为置信度，
            后四个元素为边界框 `(cx, cy, w, h)`
        """
        N = preds[0].size(0)

        # 解码
        batch_pred = []
        for pred, anchors in zip(preds, self.anchors):
            pred_ = decode(pred, anchors, self.n_classes, self.image_size)

            # 展平预测框，shape: (N, n_anchors*H*W, n_classes+5)
            batch_pred.append(pred_.view(N, -1, self.n_classes+5))

        batch_pred = torch.cat(batch_pred, dim=1)

        # 非极大值抑制
        out = []
        for pred in batch_pred:
            # 计算得分
            pred[:, 5:] = pred[:, 5:] * pred[:, 4:5]

            # 选取出类别置信度最高的那个类作为预测框的预测结果, shape: (n_anchors*H*W, 6)
            conf, c = torch.max(pred[:, 5:], dim=1, keepdim=True)
            pred = torch.cat((pred[:, :4], conf, c), dim=1)

            # 过滤掉置信度太低的预测框
            pred = pred[pred[:, 4] >= self.conf_thresh]
            if not pred.size(0):
                continue

            # 预测到的类别种类
            classes_pred = pred[:, -1].unique()

            # 对每一个类别的预测框进行筛选
            detections = {}
            for c in classes_pred:
                mask = pred[:, -1] == c
                boxes = pred[:, :4][mask]
                scores = pred[:, 4][mask]
                keep = nms(center_to_corner(boxes), scores, self.nms_thresh)
                detections[int(c)] = torch.cat(
                    (scores[keep].unsqueeze(1), boxes[keep]), dim=1)

            out.append(detections)

        return out
