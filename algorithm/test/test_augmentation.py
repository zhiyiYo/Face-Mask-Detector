# coding:utf-8
import unittest

import torch
import numpy as np

from net.dataset import VOCDataset
from utils.augmentation_utils import (BBoxToAbsoluteCoords, Compose,
                                      YoloAugmentation, ColorAugmentation)
from utils.box_utils import draw, corner_to_center_numpy


class TestAugmention(unittest.TestCase):
    """ 测试数据增强 """

    def __init__(self, methodName) -> None:
        super().__init__(methodName)
        root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
        self.dataset = VOCDataset(
            root,
            'trainval',
            color_transformer=ColorAugmentation(),
            use_mosaic=True,
            use_mixup=True
        )

    def test_voc_augmenter(self):
        """ 测试 VOC 图像增强器 """
        self.dataset.transformer = Compose(
            [YoloAugmentation(416), BBoxToAbsoluteCoords()])
        self.dataset.color_transformer = Compose([
            ColorAugmentation(), BBoxToAbsoluteCoords()])
        image, target = self.dataset[4]
        self.draw(image, target)

    def test_mosaic(self):
        """ 测试马赛克图像增强 """
        image, bbox, label = self.dataset.make_mosaic(0)
        bbox *= image.shape[0]
        image = torch.from_numpy(image).permute(2, 0, 1)/255
        bbox = corner_to_center_numpy(bbox)
        target = np.hstack((bbox, label[:, np.newaxis]))
        self.draw(image, target)

    def draw(self, image: torch.Tensor, target):
        """ 绘制图像 """
        image = image.permute(1, 2, 0).numpy()*255
        label = [self.dataset.classes[int(i)] for i in target[:, 4]]

        # 绘制边界框和标签
        image = draw(image, target[:, :4], label)
        image.show()
