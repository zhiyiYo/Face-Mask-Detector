# coding:utf-8
import unittest

import torch
from net import Yolo


class TestYolo(unittest.TestCase):
    """ 测试 Yolo 模型 """

    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        self.model = Yolo(20).cuda()

    def test_forward(self):
        """ 测试前馈 """
        x = torch.rand(2, 3, 416, 416).cuda()
        y1, y2, y3 = self.model(x)
        self.assertEqual(y1.size(), torch.Size((2, 75, 13, 13)))
        self.assertEqual(y2.size(), torch.Size((2, 75, 26, 26)))
        self.assertEqual(y3.size(), torch.Size((2, 75, 52, 52)))

    def test_predict(self):
        """ 测试推理 """
        x = torch.rand(2, 3, 416, 416).cuda()
        out = self.model.predict(x)
        print('\n预测结果：', out[0])
