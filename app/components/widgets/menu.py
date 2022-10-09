# coding:utf-8
from app.components.qframelesswindow import WindowEffect
from PyQt5.QtCore import QEvent, QFile, Qt
from PyQt5.QtWidgets import QMenu


class AeroMenu(QMenu):
    """ Aero菜单 """

    def __init__(self, string="", parent=None):
        super().__init__(string, parent)
        # 创建窗口特效
        self.windowEffect = WindowEffect(self)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Popup | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground | Qt.WA_StyledBackground)
        self.setObjectName("AeroMenu")
        self.setQss()

    def event(self, e: QEvent):
        if e.type() == QEvent.WinIdChange:
            self.setMenuEffect()
        return QMenu.event(self, e)

    def setMenuEffect(self):
        """ 开启特效 """
        self.windowEffect.setAeroEffect(self.winId())
        self.windowEffect.addShadowEffect(self.winId())

    def setQss(self):
        """ 设置层叠样式 """
        f = QFile(":/qss/menu.qss")
        f.open(QFile.ReadOnly)
        self.setStyleSheet(str(f.readAll(), encoding='utf-8'))
        f.close()


class AcrylicMenu(QMenu):
    """ 亚克力菜单 """

    def __init__(self, acrylicColor="F3F3F399", parent=None):
        super().__init__(parent=parent)
        self.acrylicColor = acrylicColor
        self.windowEffect = WindowEffect()
        self.__initWidget()

    def __initWidget(self):
        """ 初始化菜单 """
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Popup | Qt.NoDropShadowWindowHint)
        # QtWin.enableBlurBehindWindow(self)
        self.windowEffect.addMenuAnimation(self.winId())
        self.windowEffect.setAcrylicEffect(self.winId(), self.acrylicColor)

        self.setProperty("effect", "acrylic")
        self.setObjectName("acrylicMenu")
        self.setQss()

    def setQss(self):
        """ 设置层叠样式 """
        f = QFile(":/qss/menu.qss")
        f.open(QFile.ReadOnly)
        self.setStyleSheet(str(f.readAll(), encoding='utf-8'))
        f.close()


class DWMMenu(QMenu):
    """ 使用 DWM 窗口阴影的菜单 """

    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        # 创建窗口特效
        self.windowEffect = WindowEffect(self)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Popup | Qt.NoDropShadowWindowHint)
        self.setAttribute(Qt.WA_StyledBackground)
        self.setQss()

    def event(self, e: QEvent):
        if e.type() == QEvent.WinIdChange:
            self.windowEffect.addShadowEffect(self.winId())
        return QMenu.event(self, e)

    def setQss(self):
        """ 设置层叠样式表 """
        f = QFile(":/qss/menu.qss")
        f.open(QFile.ReadOnly)
        self.setStyleSheet(str(f.readAll(), encoding='utf-8'))
        f.close()
