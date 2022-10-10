# coding:utf-8
from PyQt5.QtCore import QObject, pyqtSignal
from .singleton import Singleton


class SignalBus(Singleton, QObject):
    """ Signal bus in Groove Music """
    switchToImageInterfaceSig = pyqtSignal()  # 切换到图像界面
    switchToSettingInterfaceSig = pyqtSignal()  # 切换到设置界面


signalBus = SignalBus()
