# coding:utf-8
from app.components.dialog_box import SelectModelDialog
from app.components.widgets.label import ClickableLabel
from app.components.widgets.scroll_area import ScrollArea
from app.components.widgets.slider import Slider
from app.components.widgets.switch_button import SwitchButton
from PyQt5.QtCore import QFile, Qt, pyqtSignal
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QLabel, QWidget
from torch import cuda

from app.common.config import config
from app.common.signal_bus import signalBus


class SettingInterface(ScrollArea):
    """ 设置界面 """

    enableAcrylicChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # 滚动部件
        self.scrollWidget = QWidget(self)

        # 设置标签
        self.settingLabel = QLabel(self.tr('Settings'), self)

        # 此 PC 上的热斑侦探
        self.modelInPCLabel = QLabel(
            self.tr('Face Mask Detector in this PC'), self.scrollWidget)
        self.selectModelLabel = ClickableLabel(
            self.tr('Choose where we look for Face Mask Detector'), self.scrollWidget)

        # 使用亚克力背景
        self.acrylicLabel = QLabel(
            self.tr("Acrylic Background"), self.scrollWidget)
        self.acrylicHintLabel = QLabel(
            self.tr("Use the acrylic background effect"), self.scrollWidget)
        self.acrylicSwitchButton = SwitchButton(
            self.tr("Off"), self.scrollWidget)

        # 显卡
        self.useGPULabel = QLabel(self.tr('Graphics Card'), self.scrollWidget)
        self.useGPUSwitchButton = SwitchButton(parent=self.scrollWidget)
        self.useGPUHintLabel = QLabel(
            self.tr('Use GPU to speed up Face Mask Detector thinking (if available)'), self.scrollWidget)

        # 置信度
        self.confLabel = QLabel(self.tr('Confidence'), self.scrollWidget)
        self.confHintLabel = QLabel(self.tr(
            'Set the confidence threshold. The lower the threshold, the more prediction boxes of face'), self.scrollWidget)
        self.confSlider = Slider(Qt.Horizontal, self.scrollWidget)
        self.confValueLabel = QLabel(self.scrollWidget)

        # 初始化
        self.__initWidget()

    def __initWidget(self):
        """ 初始化界面 """
        self.resize(560, 560)
        self.scrollWidget.resize(540, 560)
        self.setWidget(self.scrollWidget)
        self.setViewportMargins(0, 88, 0, 0)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 设置标签
        self.settingLabel.move(24, 44)

        # 选择模型文件
        self.modelInPCLabel.move(24, 19)
        self.selectModelLabel.move(24, 62)

        # 亚克力背景
        self.acrylicLabel.move(24, 110)
        self.acrylicHintLabel.move(24, 149)
        self.acrylicSwitchButton.move(24, 173)

        # 使用 GPU 加速
        self.useGPULabel.move(24, 221)
        self.useGPUHintLabel.move(24, 259)
        self.useGPUSwitchButton.move(24, 283)

        # 置信度阈值
        self.confLabel.move(24, 331)
        self.confHintLabel.move(24, 369)
        self.confSlider.move(24, 393)
        self.confValueLabel.move(184, 393)

        # 初始化亚克力背景开关按钮
        enableAcrylic = config.get(config.enableAcrylicBackground)
        self.acrylicSwitchButton.setText(
            self.tr('On') if enableAcrylic else self.tr('Off'))
        self.acrylicSwitchButton.setChecked(enableAcrylic)

        # 初始化 GPU 开关按钮
        isUseGPU = config.get(config.useGPU) and cuda.is_available()
        self.useGPUSwitchButton.setText(
            self.tr('On') if isUseGPU else self.tr('Off'))
        self.useGPUSwitchButton.setChecked(isUseGPU)
        self.useGPUSwitchButton.setEnabled(cuda.is_available())

        # 初始化置信度滑动条
        low, high = config.confidenceThreshold.range
        self.confSlider.setRange(low*100, high*100)
        self.confSlider.setSingleStep(1)
        self.confSlider.setValue(config.get(config.confidenceThreshold)*100)
        self.confValueLabel.setText(
            f"{config.get(config.confidenceThreshold):.2f}")

        # 设置层叠样式
        self.__setQss()

        # 信号连接到槽
        self.__connectSignalToSlot()

    def __setQss(self):
        """ 设置层叠样式 """
        self.settingLabel.setObjectName('settingLabel')
        self.useGPULabel.setObjectName('titleLabel')
        self.acrylicLabel.setObjectName('titleLabel')
        self.modelInPCLabel.setObjectName('titleLabel')
        self.confLabel.setObjectName('titleLabel')
        self.selectModelLabel.setObjectName("clickableLabel")

        f = QFile(':/qss/setting_interface.qss')
        f.open(QFile.ReadOnly)
        self.setStyleSheet(str(f.readAll(), encoding='utf-8'))
        f.close()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.scrollWidget.resize(self.width(), self.scrollWidget.height())

    def paintEvent(self, e):
        """ 绘制视口背景 """
        super().paintEvent(e)
        painter = QPainter(self.viewport())
        painter.setBrush(Qt.white)
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, 0, self.width(), self.height() -
                         self.contentsMargins().top())

    def __showSelectModelDialog(self):
        """ 显示选择模型对话框 """
        w = SelectModelDialog(config.get(config.modelPath), self.window())
        w.modelChangedSignal.connect(self.__onModelChanged)
        w.exec_()

    def __onModelChanged(self, model: str):
        """ 模型改变信号槽函数 """
        if model == config.get(config.modelPath):
            return

        config.set(config.modelPath, model)
        signalBus.modelChanged.emit(model)

    def __onEnableAcrylicChanged(self, isEnable: bool):
        """ 使用亚克力背景开关按钮的开关状态变化槽函数 """
        config.set(config.enableAcrylicBackground, isEnable)
        self.acrylicSwitchButton.setText(
            self.tr('On') if isEnable else self.tr('Off'))
        self.enableAcrylicChanged.emit(isEnable)

    def __onUseGPUChanged(self, useGPU: bool):
        """ 使用 GPU 加速开关按钮的开关状态改变槽函数 """
        config.set(config.useGPU, useGPU)
        self.useGPUSwitchButton.setText(
            self.tr('On') if useGPU else self.tr('Off'))
        signalBus.useGPUChanged.emit(useGPU)

    def __onConfThreshChanged(self, thresh: int):
        """ 调整置信度阈值槽函数 """
        thresh /= 100
        config.set(config.confidenceThreshold, thresh)
        self.confValueLabel.setText(f"{(thresh):.2f}")
        self.confValueLabel.adjustSize()

    def __connectSignalToSlot(self):
        """ 信号连接到槽 """
        self.selectModelLabel.clicked.connect(self.__showSelectModelDialog)
        self.confSlider.valueChanged.connect(self.__onConfThreshChanged)
        self.acrylicSwitchButton.checkedChanged.connect(
            self.__onEnableAcrylicChanged)
        self.useGPUSwitchButton.checkedChanged.connect(
            self.__onUseGPUChanged)
