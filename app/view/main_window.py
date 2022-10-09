# coding: utf-8
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from app.common import resource
from app.components.qframelesswindow import AcrylicWindow
from app.components.title_bar import TitleBar

from .image_viewer import ImageViewer


class MainWindow(AcrylicWindow):
    """ 主界面 """

    def __init__(self):
        super().__init__()
        self.imageViewer = ImageViewer(self)
        self.setTitleBar(TitleBar(self))
        self.titleBar.raise_()

        self.initWidget()

    def initWidget(self):
        """ 初始化小部件 """
        self.setWindowIcon(QIcon(":/images/logo.png"))
        self.setWindowTitle(self.tr("Face Mask Detector"))
