# coding: utf-8
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class SerialThread(QThread):
    """ 串口线程 """

    loadImageFinished = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        """ 将串口传输的字节转换为图像 """
        