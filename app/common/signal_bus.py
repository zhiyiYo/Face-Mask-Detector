# coding:utf-8
from PyQt5.QtCore import QObject, pyqtSignal
from .singleton import Singleton


class SignalBus(Singleton, QObject):
    """ 信号总线 """
    switchToImageInterfaceSig = pyqtSignal()  # 切换到图像界面
    switchToSettingInterfaceSig = pyqtSignal()  # 切换到设置界面

    modelChanged = pyqtSignal(str)      # 模型路径改变
    useGPUChanged = pyqtSignal(bool)    # 启用/禁用显卡加速

signalBus = SignalBus()
