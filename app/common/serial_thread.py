# coding: utf-8
from app.common.config import config
from serial import Serial
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from .image_utils import rgb565ToImage


class SerialThread(QThread):
    """ 串口线程 """

    loadImageFinished = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.serial = Serial(baudrate=1500000)
        self.isStopped = False

    def run(self):
        """ 将串口传输的字节转换为图像 """
        data = []
        self.serial.port = config.get(config.serialPort)

        with self.serial as s:
            while not self.isStopped:
                # 等待帧头
                header = s.readline()[:-1]
                if header.decode("utf-8", "replace") != "image:":
                    continue

                # 读入像素
                column_len = 320*2+1
                while len(data) < 2*320*240:
                    image_line = s.read(column_len)
                    data.extend(image_line[:-1])

                self.loadImageFinished.emit(rgb565ToImage(data))
                data.clear()

    def stop(self):
        self.isStopped = True

    def loadImage(self):
        self.isStopped = False
        self.start()
