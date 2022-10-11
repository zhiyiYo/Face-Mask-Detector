# coding:utf-8
from app.common.icon import Icon
from app.common.signal_bus import signalBus
from app.components.widgets.tooltip import Tooltip
from PyQt5.QtCore import QFile, QPoint, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (QFrame, QGraphicsDropShadowEffect, QHBoxLayout,
                             QToolButton, QWidget, QApplication)


class ToolBar(QFrame):
    """ 工具栏 """

    openPortSignal = pyqtSignal(bool)
    detectSignal = pyqtSignal(bool)
    zoomInSignal = pyqtSignal()
    zoomOutSignal = pyqtSignal()
    rotateSignal = pyqtSignal()
    saveImageSignal = pyqtSignal()
    copyImageSignal = pyqtSignal()
    exportSignal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.hBox = QHBoxLayout(self)

        # 创建按钮
        self.openPortButton = ToolBarButton(
            self.tr('open port'), ':/images/tool_bar/camera.png', (14, 14), self)
        self.detectButton = ToolBarButton(
            self.tr('detect face mask'), ':/images/tool_bar/detect.png', (14, 14), parent=self)
        self.zoomInButton = ToolBarButton(
            self.tr('zoom in'), ':/images/tool_bar/zoomIn.png', (14, 14), self)
        self.zoomOutButton = ToolBarButton(
            self.tr('zoom out'), ':/images/tool_bar/zoomOut.png', (14, 14), self)
        self.rotateButton = ToolBarButton(
            self.tr('rotate'), ':/images/tool_bar/rotate.png', parent=self)
        self.copyButton = ToolBarButton(
            self.tr('copy to clipboard'), ':/images/tool_bar/copy.png', parent=self)
        self.saveButton = ToolBarButton(
            self.tr('save image'), ':/images/tool_bar/export.png', parent=self)
        self.settingButton = ToolBarButton(
            self.tr('setting'), ':/images/tool_bar/setting.png', parent=self)

        # 初始化
        self.__initWidget()

    def __initWidget(self):
        """ 初始化小部件 """
        self.detectButton.setProperty('selected', False)
        self.__setShadowEffect()
        self.__setQss()

        # 设置布局
        self.hBox.setContentsMargins(4, 4, 4, 4)
        self.hBox.setSpacing(3)

        # 将按钮添加到布局中
        for i, button in enumerate(self.findChildren(QToolButton)):
            self.hBox.addWidget(button, 0, Qt.AlignLeft)
            if i == 1:
                self.hBox.addWidget(Seperator(self), 0, Qt.AlignLeft)

        self.adjustSize()

        # 信号连接到槽
        self.__connectSignalToSlot()

    def __setShadowEffect(self):
        effect = QGraphicsDropShadowEffect(self)
        effect.setColor(QColor(0, 0, 0, 50))
        effect.setOffset(0, 8)
        effect.setBlurRadius(40)
        self.setGraphicsEffect(effect)

    def __onDetectButtonClicked(self):
        """ 检测按钮点击槽函数 """
        isSelected = self.detectButton.property("selected")
        self.setDetectButtonSelected(not isSelected)
        self.detectSignal.emit(not isSelected)

    def setDetectButtonSelected(self, isSelected: bool):
        """ 设置检测按钮的选中状态 """
        if isSelected:
            self.detectButton.setProperty('selected', True)
            self.detectButton.setIcon(Icon(':/images/tool_bar/detect_white.png'))
        else:
            self.detectButton.setProperty('selected', False)
            self.detectButton.setIcon(Icon(':/images/tool_bar/detect.png'))

        # 更新样式
        self.detectButton.setStyle(QApplication.style())

    def __onOpenPortButtonClicked(self):
        """ 打开串口按钮点击槽函数 """
        isSelected = self.openPortButton.property("selected")
        self.setOpenPortButtonSelected(not isSelected)
        self.openPortSignal.emit(not isSelected)

    def setOpenPortButtonSelected(self, isSelected: bool):
        """ 设置打开串口按钮的选中状态 """
        if isSelected:
            self.openPortButton.setProperty('selected', True)
            self.openPortButton.setIcon(Icon(':/images/tool_bar/camera_white.png'))
        else:
            self.openPortButton.setProperty('selected', False)
            self.openPortButton.setIcon(Icon(':/images/tool_bar/camera.png'))

        # 更新样式
        self.openPortButton.setStyle(QApplication.style())

    def __setQss(self):
        """ 设置层叠样式 """
        self.detectButton.setObjectName('selectableButton')
        self.openPortButton.setObjectName('selectableButton')

        f = QFile(":/qss/tool_bar.qss")
        f.open(QFile.ReadOnly)
        self.setStyleSheet(str(f.readAll(), encoding='utf-8'))
        f.close()

    def __connectSignalToSlot(self):
        """ 信号连接到槽 """
        self.openPortButton.clicked.connect(self.__onOpenPortButtonClicked)
        self.detectButton.clicked.connect(self.__onDetectButtonClicked)
        self.zoomInButton.clicked.connect(self.zoomInSignal)
        self.zoomOutButton.clicked.connect(self.zoomOutSignal)
        self.rotateButton.clicked.connect(self.rotateSignal)
        self.copyButton.clicked.connect(self.copyImageSignal)
        self.saveButton.clicked.connect(self.saveImageSignal)
        self.settingButton.clicked.connect(
            signalBus.switchToSettingInterfaceSig)


class ToolBarButton(QToolButton):
    """ 工具栏按钮 """

    def __init__(self, text: str, iconPath: str, size=(15, 15), parent=None):
        """
        Parameters
        ----------
        text: str
            按钮文字

        iconPath: str
            图标路径

        size: tuple
            图标大小

        parent:
            父级窗口
        """
        super().__init__(parent=parent)
        self.setIcon(Icon(iconPath))
        self.setIconSize(QSize(*size))
        self.setFixedSize(40, 40)
        self.tip = Tooltip(text, self.window())
        self.tip.hide()

    def enterEvent(self, e):
        """ 鼠标进入时显示提示条 """
        if not self.isEnabled():
            return

        pos = self.mapTo(self.window(), QPoint(0, 0))
        x = pos.x() + self.width()//2 - self.tip.width()//2
        y = pos.y() - 2 - self.tip.height()
        x = min(max(5, x), self.window().width() - self.tip.width() - 5)

        self.tip.move(x, y)
        self.tip.show()

    def leaveEvent(self, e):
        """ 鼠标离开时隐藏提示条 """
        self.tip.hide()


class Seperator(QWidget):
    """ 分隔符 """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedSize(20, 40)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, e):
        """ 绘制分隔符 """
        painter = QPainter(self)
        painter.setPen(QPen(QColor(217, 217, 217), 1))
        painter.drawLine(9, 10, 9, 31)
