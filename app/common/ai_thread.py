# coding:utf-8
import os

import numpy as np
import torch
from algorithm.net import Yolo
from algorithm.net.dataset import VOCDataset
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from .config import config
from .logger import Logger

logger = Logger("AI_thread")


class AIThread(QThread):
    """ 检测口罩线程 """

    detectFinished = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.image = None
        self.detectResult = QPixmap()
        self.loadModel()

    def run(self):
        """ 检测图片中的口罩 """
        if self.model is None:
            logger.warning('当前未选择任何模型！')
            return

        # 检测图像
        self.model.detector.conf_thresh = config.get(config.confidenceThreshold)
        image = self.model.detect(
            self.image, VOCDataset.classes, use_gpu=config.get(config.useGPU))

        # 将图像转换为 QPixmap
        image = np.array(image)
        h, w, _ = image.shape
        pixmap = QPixmap.fromImage(
            QImage(image.data, w, h, 3 * w, QImage.Format_RGB888))

        self.detectFinished.emit(pixmap)

    def detect(self, pixmap: QPixmap):
        """ 检测图像 """
        if pixmap.isNull():
            return

        self.image = Image.fromqpixmap(pixmap)
        self.start()

    def loadModel(self):
        device = torch.device('cuda' if config.get(config.useGPU) else 'cpu')

        # 创建模型
        modelPath = config.get(config.modelPath)
        if not os.path.exists(modelPath) or os.path.isdir(modelPath):
            self.model = None
        else:
            anchors = [
                [[100, 146], [147, 203], [208, 260]],
                [[26, 43], [44, 65], [65, 105]],
                [[4, 8], [8, 15], [15, 27]]
            ]
            self.model = Yolo(n_classes=2, anchors=anchors).to(device)
            self.model.load(modelPath)
            self.model.eval()

    def setUseGPU(self, useGPU: bool):
        """ 设置是否启用 GPU 加速 """
        device = torch.device('cuda' if useGPU else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
