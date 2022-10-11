# coding: utf-8
from app.common import resource
from app.common.config import config
from app.common.signal_bus import signalBus
from app.components.qframelesswindow import AcrylicWindow
from app.components.title_bar import TitleBar
from app.components.widgets.pop_up_stacked_widget import PopUpStackedWidget
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication

from .image_interface import ImageInterface
from .setting_interface import SettingInterface


class MainWindow(AcrylicWindow):
    """ 主界面 """

    def __init__(self):
        super().__init__()
        self.stackWidget = PopUpStackedWidget(self)
        self.settingInterface = SettingInterface()
        self.imageInterface = ImageInterface(self)
        self.setTitleBar(TitleBar(self))

        self.initWidget()
        self.imageInterface.setImage(QPixmap(
            r"D:\C++_Study\Github_Repositories\Face_Mask_Detector\algorithm\resource\image\三上老师.jpg"))

    def initWidget(self):
        """ 初始化小部件 """
        self.resize(800, 700)
        self.setObjectName('mainWindow')
        self.setWindowIcon(QIcon(":/images/logo.png"))
        self.setWindowTitle(self.tr("Face Mask Detector"))
        self.setAcrylicEnabled(True)

        # 居中
        desktop = QApplication.desktop().availableGeometry()
        self.move(desktop.width()//2 - self.width()//2,
                  desktop.height()//2 - self.height()//2)

        # 添加界面
        self.stackWidget.addWidget(self.imageInterface, 0, 0, False)
        self.stackWidget.addWidget(self.settingInterface, 0, 120, False)

        self.titleBar.raise_()
        self.connectSignalToSlot()

    def setAcrylicEnabled(self, isEnabled: bool):
        """ 设置窗口特效 """
        if isEnabled:
            self.windowEffect.setAcrylicEffect(self.winId(), "FFFFFFE6")
            self.setStyleSheet("#mainWindow{background:transparent}")
        else:
            self.setStyleSheet("#mainWindow{background:#F3F3F3}")
            self.windowEffect.addShadowEffect(self.winId())
            self.windowEffect.removeBackgroundEffect(self.winId())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.stackWidget.resize(self.size())

    def switchToSettingInterface(self):
        """ 切换到设置界面 """
        self.stackWidget.setCurrentWidget(self.settingInterface)
        self.titleBar.setReturnButtonVisible(True)

    def switchToImageInterface(self):
        """ 切换到图像界面 """
        self.stackWidget.setCurrentWidget(self.imageInterface, True, False)
        self.titleBar.setReturnButtonVisible(False)

    def connectSignalToSlot(self):
        """ 信号连接到槽 """
        self.settingInterface.enableAcrylicChanged.connect(self.setAcrylicEnabled)
        self.titleBar.returnButton.clicked.connect(self.switchToImageInterface)
        signalBus.switchToSettingInterfaceSig.connect(self.switchToSettingInterface)
