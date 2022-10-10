# coding:utf-8
from app.common.ai_thread import AIThread
from app.components.dialog_box import Dialog
from app.components.widgets.label import PixmapLabel
from PyQt5.QtCore import QFile, QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPixmap, QTransform, QWheelEvent
from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QGraphicsItem,
                             QGraphicsPixmapItem, QGraphicsScene,
                             QGraphicsView, QLabel, QWidget)

from .tool_bar import ToolBar


class ImageViewer(QGraphicsView):
    """ 图片查看器 """

    clicked = pyqtSignal()
    showBarSignal = pyqtSignal()
    hideBarSignal = pyqtSignal()
    exitDrawModeSignal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.isDrawModeEnabled = False
        self.zoomInTimes = 0
        self.maxZoomInTimes = 22
        self.angle = 0

        # 创建场景
        self.graphicsScene = QGraphicsScene()

        # 图片
        self.pixmap = QPixmap()
        self.pixmapItem = QGraphicsPixmapItem()
        self.displayedImageSize = QSize(0, 0)

        # 快捷键
        self.zoomInAction = QAction(
            self.tr('zoom in'), self, triggered=self.zoomIn, shortcut='Ctrl++')
        self.zoomOutAction = QAction(
            self.tr('zoom out'), self, triggered=self.zoomOut, shortcut='Ctrl+-')
        self.rotateAction = QAction(
            self.tr('rotate'), self, triggered=self.rot90, shortcut='Ctrl+R')

        # 初始化小部件
        self.__initWidget()

    def __initWidget(self):
        """ 初始化小部件 """
        self.resize(1200, 900)

        # 隐藏滚动条
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 以鼠标所在位置为锚点进行缩放
        self.setTransformationAnchor(self.AnchorUnderMouse)

        # 平滑缩放
        self.pixmapItem.setTransformationMode(Qt.SmoothTransformation)
        self.setRenderHints(QPainter.Antialiasing |
                            QPainter.SmoothPixmapTransform)

        # 设置场景
        self.graphicsScene.addItem(self.pixmapItem)
        self.setScene(self.graphicsScene)

        # 添加快捷键
        self.addActions(
            [self.zoomInAction, self.zoomOutAction, self.rotateAction])

    def wheelEvent(self, e: QWheelEvent):
        """ 滚动鼠标滚轮缩放图片 """
        if e.angleDelta().y() > 0:
            if self.zoomInTimes == 0:
                self.hideBarSignal.emit()
            self.zoomIn()
        else:
            self.zoomOut()

    def resizeEvent(self, e):
        """ 缩放图片 """
        super().resizeEvent(e)

        if self.zoomInTimes > 0:
            return

        # 调整图片大小
        ratio = self.__getScaleRatio()
        self.displayedImageSize = self.pixmap.size()*ratio
        if ratio < 1:
            self.fitInView(self.pixmapItem, Qt.KeepAspectRatio)
        else:
            self.resetScale()

    def setImage(self, image: QPixmap):
        """ 设置显示的图片 """
        self.resetTransform()

        # 刷新图片
        self.pixmap = image
        self.pixmapItem.setPixmap(self.pixmap)

        # 调整图片大小
        self.setSceneRect(QRectF(self.pixmap.rect()))
        ratio = self.__getScaleRatio()
        self.displayedImageSize = self.pixmap.size()*ratio
        if ratio < 1:
            self.fitInView(self.pixmapItem, Qt.KeepAspectRatio)

    def resetTransform(self):
        """ 重置变换 """
        super().resetTransform()
        self.zoomInTimes = 0
        self.__setDragEnabled(False)

    def resetScale(self):
        """ 重置缩放 """
        self.setTransform(QTransform().scale(1, 1).rotate(self.angle))
        self.zoomInTimes = 0
        self.__setDragEnabled(False)

    def __isEnableDrag(self):
        """ 根据图片的尺寸决定是否启动拖拽功能 """
        v = self.verticalScrollBar().maximum() > 0
        h = self.horizontalScrollBar().maximum() > 0
        return v or h

    def __setDragEnabled(self, isEnabled: bool):
        """ 设置拖拽是否启动 """
        if self.isDrawModeEnabled:
            return

        self.setDragMode(
            self.ScrollHandDrag if isEnabled else self.NoDrag)

    def __getScaleRatio(self):
        """ 获取显示的图像和原始图像的缩放比例 """
        if self.pixmap.isNull():
            return 1

        # 考虑旋转图片的情况
        size = self.pixmap.size()
        if self.angle % 180 != 0:
            size.transpose()

        rw = min(1, self.width()/size.width())
        rh = min(1, self.height()/size.height())
        return min(rw, rh)

    def fitInView(self, item: QGraphicsItem, mode=Qt.KeepAspectRatio):
        """ 缩放场景使其适应窗口大小 """
        super().fitInView(item, mode)
        self.displayedImageSize = self.__getScaleRatio()*self.pixmap.size()
        self.zoomInTimes = 0
        self.__setDragEnabled(False)

    def zoomIn(self, viewAnchor=QGraphicsView.AnchorUnderMouse):
        """ 放大图像 """
        if self.zoomInTimes == self.maxZoomInTimes:
            return

        self.setTransformationAnchor(viewAnchor)

        self.zoomInTimes += 1
        self.scale(1.1, 1.1)
        self.__setDragEnabled(self.__isEnableDrag())

        # 还原 anchor
        self.setTransformationAnchor(self.AnchorUnderMouse)

    def zoomOut(self, viewAnchor=QGraphicsView.AnchorUnderMouse):
        """ 缩小图像 """
        if self.zoomInTimes == 0 and not self.__isEnableDrag():
            return

        self.setTransformationAnchor(viewAnchor)

        self.zoomInTimes -= 1

        # 原始图像的大小
        pw = self.pixmap.width()
        ph = self.pixmap.height()

        # 实际显示的图像宽度
        w = self.displayedImageSize.width()*1.1**self.zoomInTimes
        h = self.displayedImageSize.height()*1.1**self.zoomInTimes

        if pw > self.width() or ph > self.height():
            # 在窗口尺寸小于原始图像时禁止继续缩小图像比窗口还小
            if w <= self.width() and h <= self.height():
                self.fitInView(self.pixmapItem)
            else:
                self.scale(1/1.1, 1/1.1)
        else:
            # 在窗口尺寸大于图像时不允许缩小的比原始图像小
            if w <= pw:
                self.resetScale()
            else:
                self.scale(1/1.1, 1/1.1)

        self.__setDragEnabled(self.__isEnableDrag())

        # 还原 anchor
        self.setTransformationAnchor(self.AnchorUnderMouse)

    def rot90(self):
        """ 顺时针旋转 90 度 """
        self.angle = (self.angle+90) % 360

        # 计算变化矩阵
        r = self.__getScaleRatio()
        transform = QTransform().scale(r, r).rotate(self.angle)
        self.setTransform(transform)

        self.zoomInTimes = 0
        self.__setDragEnabled(False)
        self.displayedImageSize.transpose()


class ImageInterface(QWidget):
    """ 图像界面 """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.isDetectEnabled = False
        self.image = QPixmap()  # 没有绘制边界框的原始图像

        # 导航提示
        self.logo = PixmapLabel(self)
        self.loadImageLabel = PixmapLabel(self)
        self.hintLabel = QLabel(
            self.tr('Click the open serial port button to detect'), self)

        # 图像查看器和工具栏
        self.imageViewer = ImageViewer(self)
        self.toolBar = ToolBar(self)

        # 线程
        self.aiThread = AIThread(self)

        self.__initWidget()

    def __initWidget(self):
        """ 初始化小部件 """
        self.logo.setPixmap(QPixmap(':/images/logo.png').scaled(
            376, 376, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.loadImageLabel.setPixmap(
            QPixmap(':/images/image_interface/loadImage.png'))

        self.__setQss()
        self.connectSignalToSlot()

    def __setQss(self):
        """ 设置层叠样式 """
        self.hintLabel.setObjectName("hintLabel")

        f = QFile(":/qss/image_interface.qss")
        f.open(QFile.ReadOnly)
        self.setStyleSheet(str(f.readAll(), encoding='utf-8'))
        f.close()

        # 调整标签大小
        self.logo.adjustSize()
        self.hintLabel.adjustSize()
        self.loadImageLabel.adjustSize()

    def resizeEvent(self, e):
        self.imageViewer.move(0, 32)
        self.imageViewer.resize(self.width(), self.height()-40)
        w, h = self.width(), self.height()

        self.logo.move(w//2 - self.logo.width()//2,
                       h//2 - self.logo.height()//2)

        # 调整提示标签位置
        w_ = 50 + self.hintLabel.width()
        y = self.logo.y() + self.logo.pixmap().height() + 50
        self.loadImageLabel.move(w//2-w_//2, y+7)
        self.hintLabel.move(self.loadImageLabel.x()+50, y+9)

        # 调整工具栏位置
        self.toolBar.move(w//2 - self.toolBar.width()//2, 60)

    def __saveImage(self):
        """ 保存当前图片 """
        if self.hintLabel.isVisible():
            self.showHintOpenImageDialog()
            return

        path, _ = QFileDialog.getSaveFileName(
            self, self.tr('save as'), '.', 'JPG (*.jpg;*.jpeg;*.jpe;*.jiff);;PNG (*.png);;GIF (*.gif)')

        if path:
            self.imageViewer.pixmap.save(path)

    def setImage(self, image: QPixmap):
        """ 设置图像 """
        self.image = image
        self.imageViewer.setImage(image)
        self.hintLabel.hide()
        self.loadImageLabel.hide()
        if self.isDetectEnabled:
            self.aiThread.detect(image)

    def __copyImage(self):
        """ 复制当前图片到剪贴板 """
        if self.hintLabel.isVisible():
            self.showHintOpenImageDialog()
            return

        QApplication.clipboard().setPixmap(self.imageViewer.pixmap)

    def showHintOpenImageDialog(self):
        """ 显示提示打开图片的对话框 """
        title = self.tr('Are you sure to open serial port')
        content = self.tr(
            'No image for detection. Do you want to open the serial port to load image?')
        w = Dialog(title, content, self.window())
        # w.yesSignal.connect(self.openSerialPort)
        w.exec()

    def __setDetectEnabled(self, enabled: bool):
        """ 启用/禁用口罩检测槽函数 """
        self.isDetectEnabled = enabled
        if self.image and enabled:
            self.aiThread.detect(self.image)

    def connectSignalToSlot(self):
        """ 信号连接到槽 """
        self.toolBar.copyImageSignal.connect(self.__copyImage)
        self.toolBar.saveImageSignal.connect(self.__saveImage)
        self.toolBar.zoomInSignal.connect(self.imageViewer.zoomIn)
        self.toolBar.zoomOutSignal.connect(self.imageViewer.zoomOut)
        self.toolBar.rotateSignal.connect(self.imageViewer.rot90)
        self.toolBar.detectSignal.connect(self.__setDetectEnabled)

        self.aiThread.detectFinished.connect(self.imageViewer.setImage)
