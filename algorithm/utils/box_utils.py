# coding: utf-8
import math
from typing import List, Union

import cmapy
import numpy as np
import torch
from numpy import ndarray
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from math import pi


def iou(bbox1: Tensor, bbox2: Tensor):
    """ 计算两组边界框的交并比，四个坐标为 `(cx, cy, w, h)`

    Parameters
    ----------
    bbox1: Tensor of shape `(A, 4)`
        第一组边界框

    bbox2: Tensor of shape  `(B, 4)`
        第二组边界框

    Returns
    -------
    iou: Tensor of shape `(A, B)`
        交并比
    """
    A = bbox1.size(0)
    B = bbox2.size(0)

    bbox1 = center_to_corner(bbox1)
    bbox2 = center_to_corner(bbox2)

    # 将先验框和边界框真值的 xmax、ymax 以及 xmin、ymin进行广播使得维度一致，(A, B, 2)
    # 再计算 xmax 和 ymin 较小者、xmin 和 ymin 较大者，W=xmax较小-xmin较大，H=ymax较小-ymin较大
    xy_max = torch.min(bbox1[:, 2:].unsqueeze(1).expand(A, B, 2),
                       bbox2[:, 2:].unsqueeze(0).expand(A, B, 2))
    xy_min = torch.max(bbox1[:, :2].unsqueeze(1).expand(A, B, 2),
                       bbox2[:, :2].unsqueeze(0).expand(A, B, 2))

    # 计算交集面积
    inter = (xy_max-xy_min).clamp(min=0)
    inter = inter[:, :, 0]*inter[:, :, 1]

    # 计算每个矩形的面积
    area_prior = ((bbox1[:, 2]-bbox1[:, 0]) *
                  (bbox1[:, 3]-bbox1[:, 1])).unsqueeze(1).expand(A, B)
    area_bbox = ((bbox2[:, 2]-bbox2[:, 0]) *
                 (bbox2[:, 3]-bbox2[:, 1])).unsqueeze(0).expand(A, B)

    return inter/(area_prior+area_bbox-inter)


def jaccard_overlap_numpy(box: np.ndarray, boxes: np.ndarray):
    """ 计算一个边界框和多个边界框的交并比，坐标形式为 `(xmin, ymin, xmax, ymax)`

    Parameters
    ----------
    box: `~np.ndarray` of shape `(4, )`
        边界框

    boxes: `~np.ndarray` of shape `(n, 4)`
        其他边界框

    Returns
    -------
    iou: `~np.ndarray` of shape `(n, )`
        交并比
    """
    # 计算交集
    xy_max = np.minimum(boxes[:, 2:], box[2:])
    xy_min = np.maximum(boxes[:, :2], box[:2])
    inter = np.clip(xy_max-xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, 0]*inter[:, 1]

    # 计算并集
    area_boxes = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    area_box = (box[2]-box[0])*(box[3]-box[1])

    # 计算 iou
    iou = inter/(area_box+area_boxes-inter)  # type: np.ndarray
    return iou


def ciou(bbox1: Tensor, bbox2: Tensor):
    """ 计算两组边界框一一对应的 CIOU

    Parameters
    ----------
    bbox1, bbox2: Tensor of shape `(..., 4)`
        维度相同的边界框，形式为 `(cx, cy, w, h)`
    """
    # 转换到边角坐标形式
    xy_min1 = bbox1[..., [0, 1]] - bbox1[..., [2, 3]]/2
    xy_max1 = bbox1[..., [0, 1]] + bbox1[..., [2, 3]]/2
    xy_min2 = bbox2[..., [0, 1]] - bbox2[..., [2, 3]]/2
    xy_max2 = bbox2[..., [0, 1]] + bbox2[..., [2, 3]]/2

    # 计算 IOU
    xy_max = torch.min(xy_max1, xy_max2)
    xy_min = torch.max(xy_min1, xy_min2)
    inter = (xy_max-xy_min).clamp(min=0)
    inter = inter[..., 0]*inter[..., 1]
    union = bbox1[..., 2]*bbox1[..., 3] + bbox2[..., 2]*bbox2[..., 3] - inter
    iou = inter/(union+1e-7)

    # 计算中心距离
    center_distance = (torch.pow(bbox1[..., :2]-bbox2[..., :2], 2)).sum(dim=-1)

    # 计算对角线距离
    xy_max = torch.max(xy_max1, xy_max2)
    xy_min = torch.min(xy_min1, xy_min2)
    diag_distance = torch.pow(xy_max-xy_min, 2).sum(dim=-1)

    # 计算尺度相似性
    v = 4 / (pi**2) * torch.pow(
        torch.atan(bbox1[..., 2]/bbox1[..., 3].clamp(min=1e-6)) -
        torch.atan(bbox2[..., 2]/bbox2[..., 3].clamp(min=1e-6)), 2
    )
    alpha = v / torch.clamp((1 - iou + v), min=1e-6)

    return iou - center_distance/diag_distance.clamp(min=1e-6) - alpha*v


def center_to_corner(boxes: Tensor):
    """ 将 `(cx, cy, w, h)` 形式的边界框变换为 `(xmin, ymin, xmax, ymax)` 形式的边界框

    Parameters
    ----------
    boxes: Tensor of shape `(n, 4)`
        边界框
    """
    return torch.cat((boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2), dim=1)


def center_to_corner_numpy(boxes: ndarray) -> ndarray:
    """ 将 `(cx, cy, w, h)` 形式的边界框变换为 `(xmin, ymin, xmax, ymax)` 形式的边界框

    Parameters
    ----------
    boxes: `~np.ndarray` of shape `(n, 4)`
        边界框
    """
    return np.hstack((boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2))


def corner_to_center(boxes: Tensor):
    """ 将 `(xmin, ymin, xmax, ymax)` 形式的边界框变换为 `(cx, cy, w, h)` 形式的边界框

    Parameters
    ----------
    boxes: Tensor of shape `(n, 4)`
        边界框
    """
    return torch.cat(((boxes[:, :2]+boxes[:, 2:])/2, boxes[:, 2:]-boxes[:, :2]), dim=1)


def corner_to_center_numpy(boxes: ndarray) -> ndarray:
    """ 将 `(xmin, ymin, xmax, ymax)` 形式的边界框变换为 `(cx, cy, w, h)` 形式的边界框

    Parameters
    ----------
    boxes: `~np.ndarray` of shape `(n, 4)`
        边界框
    """
    return np.hstack(((boxes[:, :2]+boxes[:, 2:])/2, boxes[:, 2:]-boxes[:, :2]))


def decode(pred: Tensor, anchors: Union[List[List[int]], np.ndarray], n_classes: int, image_size: int, scale=True):
    """ 解码出预测框

    Parameters
    ----------
    pred: Tensor of shape `(N, (n_classes+5)*n_anchors, H, W)`
        神经网络输出的一个特征图

    anchors: List[List[int]] or `np.ndarray` of shape `(n_anchors, 2)`
        未根据特征图大小进行缩放的先验框

    n_classes: int
        类别数

    image_size: int
        输入神经网络的图像尺寸

    scale: bool
        是否将预测框缩放到原始图像的尺度，`False` 则缩放到特征图的尺度

    Returns
    -------
    out: Tensor of shape `(N, n_anchors, H, W, n_classes+5)`
        解码结果
    """
    n_anchors = len(anchors)
    N, _, h, w = pred.shape

    # 调整特征图尺寸，方便索引，调整后维度为 (N, n_anchors, H, W, n_classes+5)
    pred = pred.view(N, n_anchors, n_classes+5, h,
                     w).permute(0, 1, 3, 4, 2).contiguous().cpu()

    # 缩放先验框
    step_h = image_size/h
    step_w = image_size/w
    anchors = [[i/step_w, j/step_h] for i, j in anchors]
    anchors = Tensor(anchors)  # type:Tensor

    # 广播
    cx = torch.linspace(0, w-1, w).repeat(N, n_anchors, h, 1)
    cy = torch.linspace(0, h-1, h).view(h, 1).repeat(N, n_anchors, 1, w)
    pw = anchors[:, 0].view(n_anchors, 1, 1).repeat(N, 1, h, w)
    ph = anchors[:, 1].view(n_anchors, 1, 1).repeat(N, 1, h, w)

    # 解码
    out = torch.zeros_like(pred)
    out[..., 0] = cx + pred[..., 0].sigmoid()
    out[..., 1] = cy + pred[..., 1].sigmoid()
    out[..., 2] = pw*torch.exp(pred[..., 2])
    out[..., 3] = ph*torch.exp(pred[..., 3])
    out[..., 4:] = pred[..., 4:].sigmoid()

    # 缩放预测框到图像大小为 (image_size, image_size) 时的绝对大小
    if scale:
        out[..., [0, 2]] *= step_w
        out[..., [1, 3]] *= step_h

    return out


def match(anchors: list, anchor_mask: list, targets: List[Tensor], h: int, w: int, n_classes: int, overlap_thresh=0.5):
    """ 匹配先验框和边界框真值，标记出正负样本

    Parameters
    ----------
    anchors: list of shape `(n_anchors*3, 2)`
        根据特征图的大小进行过缩放的先验框

    anchor_mask: List[int] of shape `(n_anchors, )`
        特征图对应的先验框索引

    targets: List[Tensor] of shape (N, (n_objects, 4))
        多张图片的标签，最后一维为 `(cx, cy, w, h, class)`

    h: int
        特征图的高度

    w: int
        特征图的宽度

    n_classes: int
        类别数

    overlap_thresh: float
        IOU 阈值

    Returns
    -------
    p_mask: Tensor of shape `(N, n_anchors, H, W)`
        正例遮罩

    n_mask: Tensor of shape `(N, n_anchors, H, W)`
        反例遮罩

    gt: Tensor of shape `(N, n_anchors, H, W, n_classes+5)`
        标签，最后一个维度为 `(cx, cy, w, h, obj, c1, c2, ...)`
    """
    N = len(targets)
    n_anchors = len(anchor_mask)

    # 初始化返回值
    p_mask = torch.zeros(N, n_anchors, h, w)
    n_mask = torch.ones(N, n_anchors, h, w)
    gt = torch.zeros(N, n_anchors, h, w, n_classes+5)

    # 匹配先验框和边界框
    anchors = torch.hstack((torch.zeros((len(anchors), 2)), Tensor(anchors)))

    for i in range(N):
        if len(targets[i]) == 0:
            continue

        # 反归一化边界框
        target = torch.zeros_like(targets[i])  # shape:(n_objects, 5)
        target[:, [0, 2]] = targets[i][:, [0, 2]] * w
        target[:, [1, 3]] = targets[i][:, [1, 3]] * h
        target[:, 4] = targets[i][:, 4]
        bbox = torch.cat((torch.zeros(target.size(0), 2), target[:, 2:4]), 1)

        # 计算边界框和所有先验框的交并比
        best_indexes = torch.argmax(iou(bbox, anchors), dim=1)

        # 迭代每一个 ground truth box
        for j, best_i in enumerate(best_indexes):
            if best_i not in anchor_mask:
                continue

            k = anchor_mask.index(best_i)

            # 获取标签数据
            cx, gw = target[j, [0, 2]]
            cy, gh = target[j, [1, 3]]

            # 获取边界框中心所处的单元格的坐标
            gj, gi = int(cx), int(cy)

            # 标记出正例和反例
            p_mask[i, k, gi, gj] = 1
            # 正例除外，与 ground truth 的交并比都小于阈值则为负例
            n_mask[i, k, gi, gj] = 0
            # n_mask[i, iou >= overlap_thresh, gi, gj] = 0

            # 计算标签值
            gt[i, k, gi, gj, 0] = cx
            gt[i, k, gi, gj, 1] = cy
            gt[i, k, gi, gj, 2] = gw
            gt[i, k, gi, gj, 3] = gh
            gt[i, k, gi, gj, 4] = 1
            gt[i, k, gi, gj, 5+int(target[j, 4])] = 1

    return p_mask, n_mask, gt


def nms(boxes: Tensor, scores: Tensor, overlap_thresh=0.45, top_k=100):
    """ 非极大值抑制，去除多余的预测框

    Parameters
    ----------
    boxes: Tensor of shape `(n_boxes, 4)`
        预测框，坐标形式为 `(cx, cy, w, h)`

    scores: Tensor of shape `(n_boxes, )`
        每个预测框的置信度

    overlap_thresh: float
        IOU 阈值，大于阈值的部分预测框会被移除，值越大保留的框越多

    top_k: int
        保留的预测框个数上限

    Returns
    -------
    indexes: LongTensor of shape `(n, )`
        保留的边界框的索引
    """
    keep = []
    if boxes.numel() == 0:
        return torch.LongTensor(keep)

    # 每个预测框的面积
    boxes = center_to_corner(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2-x1)*(y2-y1)

    # 对分数进行降序排序并截取前 top_k 个索引
    _, indexes = scores.sort(dim=0, descending=True)
    indexes = indexes[:top_k]

    while indexes.numel():
        i = indexes[0]
        keep.append(i)

        # 最后一个索引时直接退出循环
        if indexes.numel() == 1:
            break

        # 其他的预测框和当前预测框的交集
        right = x2[indexes].clamp(max=x2[i].item())
        left = x1[indexes].clamp(min=x1[i].item())
        bottom = y2[indexes].clamp(max=y2[i].item())
        top = y1[indexes].clamp(min=y1[i].item())
        inter = ((right-left)*(bottom-top)).clamp(min=0)

        # 计算 iou
        iou = inter/(area[i]+area[indexes]-inter)

        # 保留 iou 小于阈值的边界框，自己和自己的 iou 为 1
        indexes = indexes[iou < overlap_thresh]

    return torch.LongTensor(keep)


def draw(image: Union[ndarray, Image.Image], bbox: ndarray, label: ndarray, conf: ndarray = None) -> Image.Image:
    """ 在图像上绘制边界框和标签

    Parameters
    ----------
    image: `~np.ndarray` of shape `(H, W, 3)` or `~PIL.Image.Image`
        RGB 图像

    bbox: `~np.ndarray` of shape `(n_objects, 4)`
        边界框，坐标形式为 `(cx, cy, w, h)`

    label: Iterable of shape `(n_objects, )`
        标签

    conf: Iterable of shape `(n_objects, )`
        置信度
    """
    bbox = center_to_corner_numpy(bbox).astype(np.int)

    if isinstance(image, ndarray):
        image = Image.fromarray(image.astype(np.uint8))  # type:Image.Image

    image_draw = ImageDraw.Draw(image, 'RGBA')
    font = ImageFont.truetype('resource/font/msyh.ttc', size=13)

    label_unique = np.unique(label).tolist()
    color_indexes = np.linspace(0, 255, len(label_unique), dtype=int)

    for i in range(bbox.shape[0]):
        x1 = max(0, bbox[i, 0])
        y1 = max(0, bbox[i, 1])
        x2 = min(image.width-1, bbox[i, 2])
        y2 = min(image.height-1, bbox[i, 3])

        # 选择颜色
        class_index = label_unique.index(label[i])
        color = to_hex_color(cmapy.color(
            'rainbow', color_indexes[class_index], True))

        # 绘制方框
        image_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 绘制标签
        y1_ = y1 if y1-23 < 0 else y1-23
        y2_ = y1 if y1_ < y1 else y1+23
        text = label[i] if conf is None else f'{label[i]} | {conf[i]:.2f}'
        l = font.getlength(text) + 3
        right = x1+l if x1+l <= image.width-1 else image.width-1
        left = int(right - l)
        image_draw.rectangle([left, y1_, right, y2_],
                             fill=color+'AA', outline=color+'DD')
        image_draw.text([left+2, y1_+2], text=text,
                        font=font, embedded_color=color)

    return image


def to_hex_color(color):
    """ 将颜色转换为 16 进制 """
    color = [hex(c)[2:].zfill(2) for c in color]
    return '#'+''.join(color)


def rescale_bbox(bbox: ndarray, image_size: int, h: int, w: int):
    """ 还原被填充和缩放后的图片的预测框

    Parameters
    ----------
    bbox: `~np.ndarray` of shape `(n_objects, 4)`
        预测框，坐标形式为 `(cx, cy, w, h)`

    image_size: int
        图像被缩放后的尺寸

    h: int
        原始图像的高度

    w: int
        原始图像的宽度

    Returns
    -------
    bbox: `~np.ndarray` of shape `(n_objects, 4)`
        预测框，坐标形式为 `(cx, cy, w, h)`
    """
    # 图像填充区域大小
    pad_x = max(h-w, 0)*image_size/max(h, w)
    pad_y = max(w-h, 0)*image_size/max(h, w)

    # 被缩放后的图像中的有效图像区域
    w_ = image_size - pad_x
    h_ = image_size - pad_y

    # 还原边界框
    bbox = center_to_corner_numpy(bbox)
    bbox[:, [0, 2]] = (bbox[:, [0, 2]]-pad_x/2)*w/w_
    bbox[:, [1, 3]] = (bbox[:, [1, 3]]-pad_y/2)*h/h_
    bbox = corner_to_center_numpy(bbox)
    return bbox
