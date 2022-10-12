# coding: utf-8
from app.common.config import config
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from serial import Serial

from .exception_handler import exceptionHandler
from .image_utils import rgb565ToImage


class SerialThread(QThread):
    """ 串口线程 """

    loadImageFinished = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.serial = Serial(baudrate=1500000)
        self.isStopped = False
        self.warn = False

    @exceptionHandler('serial')
    def run(self):
        """ 将串口传输的字节转换为图像 """
        data = []
        self.serial.port = config.get(config.serialPort)

        with self.serial as s:
            while not self.isStopped:
                if not s.isOpen():
                    s.open()

                # 发送报警信号
                # s.write(b'1' if self.warn else b'0')

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
        self.serial.close()

    def loadImage(self):
        self.isStopped = False
        self.start()

    @exceptionHandler('serial')
    def sendWarn(self, warn: bool):
        """ 发送警告 """
        self.warn = warn
        if not self.serial.isOpen():
            self.serial.open()

        self.serial.write(b'1' if warn else b'0')
