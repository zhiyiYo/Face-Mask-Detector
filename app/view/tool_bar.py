# coding:utf-8
import math

from app.components.widgets.menu import AcrylicMenu
from app.components.widgets.tooltip import Tooltip
from PyQt5.QtCore import QFile, QPoint, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen
from PyQt5.QtWidgets import (QAction, QApplication, QFrame,
                             QGraphicsDropShadowEffect, QHBoxLayout,
                             QToolButton, QWidget)


class ToolBar(QFrame):
    """ 工具栏 """

    openImageSignal = pyqtSignal()
    openFolderSignal = pyqtSignal()
    zoomInSignal = pyqtSignal()
    zoomOutSignal = pyqtSignal()
    rotateSignal = pyqtSignal()
    detectSignal = pyqtSignal()
    showInfoSignal = pyqtSignal()
    hideInfoSignal = pyqtSignal()
    boxVisibleChanged = pyqtSignal(bool)
    saveCurrentImageSignal = pyqtSignal()
    saveAllImageSignal = pyqtSignal()
    copyImageSignal = pyqtSignal()
    exportSignal = pyqtSignal()
    editModeEnabledChanged = pyqtSignal(bool)
    drawModeEnabledChanged = pyqtSignal(bool)
    switchToSettingInterfaceSig = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.isDrawModeEnabled = False
        self.isEditModeEnabled = False
        self.isInfoButtonSelected = False
        self.isBoxVisible = True
        self.hBox = QHBoxLayout(self)

        # 创建按钮
        self.openImageButton = ToolBarButton(
            self.tr('open image'), ':/images/tool_bar/photo.png', (17, 17), self)
        self.openFolderButton = ToolBarButton(
            self.tr('open folder'), ':/images/tool_bar/folder.svg', (18, 18), self)
        self.zoomInButton = ToolBarButton(
            self.tr('zoom in'), ':/images/tool_bar/zoomIn.png', (17, 17), self)
        self.zoomOutButton = ToolBarButton(
            self.tr('zoom out'), ':/images/tool_bar/zoomOut.png', (17, 17), self)
        self.rotateButton = ToolBarButton(
            self.tr('rotate'), ':/images/tool_bar/rotate.png', parent=self)
        self.editButton = ToolBarButton(
            self.tr('edit prediction box'), ':/images/tool_bar/pencil.png', parent=self)
        self.drawButton = ToolBarButton(
            self.tr('draw bounding box'), ':/images/tool_bar/draw.png', parent=self)
        self.detectButton = ToolBarButton(
            self.tr('detect hotspot'), ':/images/tool_bar/location.png', (18, 18), parent=self)
        self.viewButton = ToolBarButton(
            self.tr('show boxes'), ':/images/tool_bar/unview.png', (18, 18), parent=self)
        self.infoButton = ToolBarButton(
            self.tr('image info'), ':/images/tool_bar/tag.png', (18, 18), self)
        self.moreButton = ToolBarButton(
            self.tr('more actions'), ':/images/tool_bar/more.png', parent=self)
        self.buttons = [i for i in self.findChildren(ToolBarButton)]
        self.hiddenButtons = []

        # 创建动作
        self.openImageAction = QAction(
            QIcon(':/images/tool_bar/photo.png'), self.tr('open image'), self)
        self.openFolderAction = QAction(
            QIcon(':/images/tool_bar/folder.png'), self.tr('open folder'), self)
        self.zoomInAction = QAction(
            QIcon(':/images/tool_bar/zoomIn.png'), self.tr('zoom in'), self)
        self.zoomOutAction = QAction(
            QIcon(':/images/tool_bar/zoomOut.png'), self.tr('zoom out'), self)
        self.rotateAction = QAction(
            QIcon(':/images/tool_bar/rotate.png'), self.tr('rotate'), self)
        self.editAction = QAction(
            QIcon(':/images/tool_bar/pencil.png'), self.tr('edit prediction box'), self)
        self.drawAction = QAction(
            QIcon(':/images/tool_bar/draw.png'), self.tr('draw bounding box'), self)
        self.detectAction = QAction(
            QIcon(':/images/tool_bar/location.png'), self.tr('detect hotspot'), self)
        self.viewAction = QAction(
            QIcon(':/images/tool_bar/view.png'), self.tr('show boxes'), self)
        self.infoAction = QAction(
            QIcon(':/images/tool_bar/info.png'), self.tr('image info'), self)
        self.actions_ = [i for i in self.findChildren(QAction)]
        self.moreActions = []

        # 初始化
        self.__initWidget()

    def __initWidget(self):
        """ 初始化小部件 """
        self.__setShadowEffect()
        self.__setQss()

        # 设置布局
        self.hBox.setContentsMargins(5, 5, 5, 5)
        self.hBox.setSpacing(4)

        # 将按钮添加到布局中
        for i, button in enumerate(self.findChildren(QToolButton)):
            self.hBox.addWidget(button, 0, Qt.AlignLeft)
            if i == 1:
                self.hBox.addWidget(Seperator(self), 0, Qt.AlignLeft)

        self.hideButtons()
        self.adjustSize()

        # 信号连接到槽
        self.__connectSignalToSlot()

    def __setShadowEffect(self):
        effect = QGraphicsDropShadowEffect(self)
        effect.setColor(QColor(0, 0, 0, 50))
        effect.setOffset(-5, 10)
        effect.setBlurRadius(50)
        self.setGraphicsEffect(effect)

    def __setQss(self):
        """ 设置层叠样式 """
        self.infoButton.setObjectName('selectableButton')
        self.editButton.setObjectName('selectableButton')
        self.drawButton.setObjectName('selectableButton')
        self.infoButton.setProperty('selected', 'false')
        self.editButton.setProperty('selected', 'false')
        self.drawButton.setProperty('selected', 'false')

        f = QFile(":/qss/tool_bar.qss")
        f.open(QFile.ReadOnly)
        self.setStyleSheet(str(f.readAll(), encoding='utf-8'))
        f.close()

    def __showMoreMenu(self):
        """ 显示更多菜单 """
        menu = MoreMenu(self)

        # 将被隐藏的按钮作为菜单项插入更多菜单
        menu.insertActions(menu.saveAction, self.moreActions)
        N = 5 + len(self.moreActions)
        menu.setFixedHeight(38 + 35*N + 15*(N-1))

        # 菜单信号连接到槽
        menu.settingAction.triggered.connect(self.switchToSettingInterfaceSig)
        menu.saveAction.triggered.connect(self.saveCurrentImageSignal)
        menu.saveToFolderAction.triggered.connect(self.saveAllImageSignal)
        menu.copyAction.triggered.connect(self.copyImageSignal)
        menu.exportAction.triggered.connect(self.exportSignal)

        # 显示菜单
        pos = self.mapToGlobal(self.moreButton.pos())
        x = pos.x() + 57 - 310
        y = pos.y() + 58
        menu.exec_(QPoint(x, y))

    def __onInfoButtonClicked(self):
        """ 信息按钮点击槽函数 """
        self.setInfoButtonSelected(not self.isInfoButtonSelected)

        # 发送信号+
        if self.isInfoButtonSelected:
            self.showInfoSignal.emit()
        else:
            self.hideInfoSignal.emit()

    def __onViewButtonClicked(self):
        """ 查看边界框按钮点击槽函数 """
        self.setBoxVisible(not self.isBoxVisible)

        # 发送信号
        self.boxVisibleChanged.emit(self.isBoxVisible)

    def __onEditButtonClicked(self):
        """ 编辑按钮点击槽函数 """
        self.setEditModeEnabled(not self.isEditModeEnabled)

        # 进入编辑模式时显示边界框
        if self.isEditModeEnabled and not self.isBoxVisible:
            self.viewButton.click()

        # 发送信号
        self.editModeEnabledChanged.emit(self.isEditModeEnabled)

    def __onDrawButtonClicked(self):
        """ 绘制边界框按钮点击槽函数 """
        self.setDrawModeEnabled(not self.isDrawModeEnabled)

        # 进入绘图模式时显示边界框
        if self.isDrawModeEnabled and not self.isBoxVisible:
            self.viewButton.click()

        # 发送信号
        self.drawModeEnabledChanged.emit(self.isDrawModeEnabled)

    def setEditModeEnabled(self, isEnabled: bool):
        """ 设置编辑模式是否开启 """
        self.isEditModeEnabled = isEnabled

        if isEnabled:
            self.editButton.setProperty('selected', 'true')
            self.editButton.setIcon(QIcon(':/images/tool_bar/pencil_white.png'))
            self.editAction.setText(self.tr('exit edit mode'))
        else:
            self.editButton.setProperty('selected', 'false')
            self.editButton.setIcon(QIcon(':/images/tool_bar/pencil.png'))
            self.editAction.setText(self.tr('edit prediction box'))

        # 更新样式
        self.editButton.setStyle(QApplication.style())

    def setDrawModeEnabled(self, isEnabled: bool):
        """ 设置绘画模式是否开启 """
        self.isDrawModeEnabled = isEnabled

        if isEnabled:
            self.drawButton.setProperty('selected', 'true')
            self.drawButton.setIcon(QIcon(':/images/tool_bar/draw_white.png'))
            self.drawAction.setText(self.tr('exit draw mode'))
        else:
            self.drawButton.setProperty('selected', 'false')
            self.drawButton.setIcon(QIcon(':/images/tool_bar/draw.png'))
            self.drawAction.setText(self.tr('draw bounding box'))

        # 更新样式
        self.drawButton.setStyle(QApplication.style())

    def setBoxVisible(self, isVisible: bool):
        """ 设置边界框是否可见 """
        self.isBoxVisible = isVisible

        if isVisible:
            self.viewAction.setText(self.tr('hide boxes'))
            self.viewAction.setIcon(QIcon(':/images/tool_bar/unview.png'))
            self.viewButton.setIcon(QIcon(':/images/tool_bar/unview.png'))
            self.viewButton.tip.setText(self.tr('hide boxes'))
        else:
            self.viewAction.setText(self.tr('show boxes'))
            self.viewAction.setIcon(QIcon(':/images/tool_bar/view.png'))
            self.viewButton.setIcon(QIcon(':/images/tool_bar/view.png'))
            self.viewButton.tip.setText(self.tr('show boxes'))

    def setInfoButtonSelected(self, isSelected: bool):
        """ 设置信息按钮的选中状态 """
        self.isInfoButtonSelected = isSelected

        if isSelected:
            self.infoButton.setProperty('selected', 'true')
            self.infoButton.setIcon(QIcon(':/images/tool_bar/tag_white.png'))
        else:
            self.infoButton.setProperty('selected', 'false')
            self.infoButton.setIcon(QIcon(':/images/tool_bar/tag.png'))

        # 更新样式
        self.infoButton.setStyle(QApplication.style())

    def __connectSignalToSlot(self):
        """ 信号连接到槽 """
        # 按钮点击
        self.openImageButton.clicked.connect(self.openImageSignal)
        self.openFolderButton.clicked.connect(self.openFolderSignal)
        self.zoomInButton.clicked.connect(self.zoomInSignal)
        self.zoomOutButton.clicked.connect(self.zoomOutSignal)
        self.rotateButton.clicked.connect(self.rotateSignal)
        self.editButton.clicked.connect(self.__onEditButtonClicked)
        self.drawButton.clicked.connect(self.__onDrawButtonClicked)
        self.detectButton.clicked.connect(self.detectSignal)
        self.viewButton.clicked.connect(self.__onViewButtonClicked)
        self.infoButton.clicked.connect(self.__onInfoButtonClicked)
        self.moreButton.clicked.connect(self.__showMoreMenu)

        # 动作触发
        self.openImageAction.triggered.connect(self.openImageSignal)
        self.openFolderAction.triggered.connect(self.openFolderSignal)
        self.zoomInAction.triggered.connect(self.zoomInSignal)
        self.zoomOutAction.triggered.connect(self.zoomOutSignal)
        self.rotateAction.triggered.connect(self.rotateSignal)
        self.editAction.triggered.connect(self.__onEditButtonClicked)
        self.drawAction.triggered.connect(self.__onDrawButtonClicked)
        self.detectAction.triggered.connect(self.detectSignal)
        self.infoAction.triggered.connect(self.__onInfoButtonClicked)
        self.viewAction.triggered.connect(self.__onViewButtonClicked)

    def adjustWidth(self, width: int):
        """ 自适应调整宽度 """
        # 如果父级宽度够大且当前没有隐藏的按钮就直接返回
        if width >= 675 and not self.hiddenButtons:
            return

        # 计算需要隐藏的按钮个数
        N = math.ceil((675-width)/50)
        if N > 0:
            hiddenButtons = self.buttons[-N-1:-1]
            self.moreActions = self.actions_[-N:]

            # 显示之前可能被隐藏的按钮
            for button in self.hiddenButtons[:-N]:
                button.show()

            # 隐藏按钮
            for button in hiddenButtons:
                button.hide()

            self.hiddenButtons = hiddenButtons

        elif width >= 675:
            for button in self.hiddenButtons:
                button.show()

            self.hiddenButtons.clear()
            self.moreActions.clear()

        self.hideButtons()
        self.adjustSize()

    def hideButtons(self):
        self.zoomInButton.hide()
        self.zoomOutButton.hide()
        self.rotateButton.hide()


class ToolBarButton(QToolButton):
    """ 工具栏按钮 """

    def __init__(self, text: str, iconPath: str, size=(19, 19), parent=None):
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
        self.setIcon(QIcon(iconPath))
        self.setIconSize(QSize(*size))
        self.setFixedSize(50, 50)
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
        self.setFixedSize(25, 50)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, e):
        """ 绘制分隔符 """
        painter = QPainter(self)
        painter.setPen(QPen(QColor(217, 217, 217), 1))
        painter.drawLine(12, 13, 12, 38)


class MoreMenu(AcrylicMenu):
    """ 更多菜单 """

    def __init__(self, parent=None):
        super().__init__('F3F3F3E3', parent=parent)
        self.saveAction = QAction(
            QIcon(':/images/tool_bar/save.png'), self.tr('save as'), self)
        self.copyAction = QAction(
            QIcon(':/images/tool_bar/copy.png'), self.tr('copy to clipboard'), self)
        self.saveToFolderAction = QAction(
            QIcon(':/images/tool_bar/saveToDir.svg'), self.tr('save to directory'), self)
        self.exportAction = QAction(
            QIcon(':/images/tool_bar/export.png'), self.tr('export'), self)
        self.settingAction = QAction(
            QIcon(':/images/tool_bar/setting.png'), self.tr('setting'), self)

        self.addActions([
            self.saveAction, self.copyAction, self.saveToFolderAction,
            self.exportAction, self.settingAction
        ])

        self.setFixedSize(321, 38+35*5+15*4)
