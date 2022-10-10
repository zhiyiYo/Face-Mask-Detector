# coding: utf-8
from app.common.icon import Icon
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel

from ..qframelesswindow import TitleBar as TB
from ..qframelesswindow.titlebar import CloseButton
from ..widgets.label import PixmapLabel


class ReturnButton(CloseButton):
    """ 返回按钮 """

    def __init__(self, parent=None):
        defaultStyle = {
            "normal": {
                'background': (0, 107, 131),
                "icon": ":/images/title_bar/返回按钮_normal_60_40.png"
            },
            "hover": {
                'background': (43, 116, 131),
                "icon": ":/images/title_bar/返回按钮_hover_60_40.png"
            },
            "pressed": {
                'background': (95, 137, 147),
                "icon": ":/images/title_bar/返回按钮_pressed_60_40.png"
            },
        }
        super().__init__(defaultStyle, parent)
        self.setIconSize(QSize(48, 32))
        self.setIcon(Icon(self._style['normal']['icon']))


class TitleBar(TB):
    """ 标题栏 """

    def __init__(self, parent):
        super().__init__(parent)
        self.logo = PixmapLabel(self)
        self.titleLabel = QLabel(self.tr("Face Mask Detector"), self)
        self.returnButton = ReturnButton(self)
        self.__initWidget()

    def __initWidget(self):
        """ 初始化小部件 """
        self.logo.setPixmap(QPixmap(':/images/logo.png').scaled(20,
                            20, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo.adjustSize()
        self.titleLabel.setStyleSheet(
            "QLabel{font: 12px 'Segoe UI Semilight', 'Microsoft YaHei Light'}")

        self.setReturnButtonVisible(False)

    def setReturnButtonVisible(self, isVisible: bool):
        """ 设置返回按钮可见性 """
        self.returnButton.setVisible(isVisible)
        self.logo.move(isVisible*48+10, 7)
        self.titleLabel.move(self.logo.pixmap().width()+self.logo.x()+2, 2)
