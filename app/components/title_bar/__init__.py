# coding: utf-8
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel

from ..qframelesswindow import TitleBar as TB
from ..widgets.label import PixmapLabel


class TitleBar(TB):
    """ 标题栏 """

    def __init__(self, parent):
        super().__init__(parent)
        self.logo = PixmapLabel(self)
        self.titleLabel = QLabel(self.tr("Face Mask Detector"), self)

        self.logo.setPixmap(QPixmap(':/images/logo.png').scaled(20,
                            20, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo.adjustSize()
        self.logo.move(10, 5)
        self.titleLabel.move(self.logo.pixmap().width()+self.logo.x()+2, 0)
        self.titleLabel.setStyleSheet(
            "QLabel{font: 12px 'Segoe UI Semilight', 'Microsoft YaHei Light'}")
