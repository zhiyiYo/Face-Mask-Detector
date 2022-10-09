# coding:utf-8
from PyQt5.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import (QPainter, QPixmap, QTransform,
                         QWheelEvent)
from PyQt5.QtWidgets import (QAction, QGraphicsItem, QGraphicsPixmapItem,
                             QGraphicsScene, QGraphicsView)


class ImageViewer(QGraphicsView):
    """ 图片查看器 """

    clicked = pyqtSignal()
    showBarSignal = pyqtSignal()
    hideBarSignal = pyqtSignal()
    exitDrawModeSignal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.drawStartPos = None
        self.isEditModeEnabled = False
        self.isDrawModeEnabled = False
        self.isRectItemVisible = False
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

    def setImage(self, imagePath: str):
        """ 设置显示的图片 """
        self.resetTransform()
        self.clearRectItems()

        # 刷新图片
        self.pixmap = QPixmap(imagePath)
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

