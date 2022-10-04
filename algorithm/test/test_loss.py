# coding:utf-8
import unittest
import numpy as np
import torch

from algorithm.net.loss import YoloLoss
from algorithm.net.yolo import Yolo


class TestLoss(unittest.TestCase):
    """ 测试损失函数 """

    def test_loss(self):
        """ 测试损失 """
        anchors = [
            [[142, 110], [192, 243], [459, 401]],
            [[36, 75], [76, 55], [72, 146]],
            [[12, 16], [19, 36], [40, 28]],
        ]
        yolo_loss = YoloLoss(anchors, 20, 416)
        model = Yolo(20, 416, anchors)

        pred = model(torch.rand(2, 3, 416, 416))
        target = [
            torch.Tensor(np.array([
                [0, 0.2, 0.3, 0.3, 0.4],
                [1, 0.4, 0.3, 0.2, 0.3]
            ])),
            torch.Tensor(np.array([
                [5, 0.1, 0.5, 0.4, 0.3],
                [3, 0.6, 0.4, 0.2, 0.5]
            ])),
        ]
        loc_loss, conf_loss, cls_loss = yolo_loss(pred, target)
        print('\nloc_loss=', loc_loss)
        print('conf_loss=', conf_loss)
        print('cls_loss=', cls_loss)