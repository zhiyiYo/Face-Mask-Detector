# coding:utf-8
from app.common.auto_wrap import autoWrap
from PyQt5.QtCore import QFile, Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import QDialog, QLabel, QPushButton


class Dialog(QDialog):

    yesSignal = pyqtSignal()
    cancelSignal = pyqtSignal()

    def __init__(self, title: str, content: str, parent=None):
        super().__init__(parent, Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        self.resize(240, 192)
        self.setWindowTitle(title)
        self.content = content
        self.titleLabel = QLabel(title, self)
        self.contentLabel = QLabel(content, self)
        self.yesButton = QPushButton(self.tr('OK'), self)
        self.cancelButton = QPushButton(self.tr('Cancel'), self)
        self.__initWidget()

    def __initWidget(self):
        """ 初始化小部件 """
        self.yesButton.setFocus()
        self.titleLabel.move(24, 24)
        self.contentLabel.move(24, 60)
        self.contentLabel.setText(autoWrap(self.content, 68)[0])

        # 设置层叠样式
        self.__setQss()

        # 调整窗口大小
        rect = self.contentLabel.geometry()
        self.setFixedSize(48 + rect.width(), rect.bottom() + 105)

        # 信号连接到槽
        self.yesButton.clicked.connect(self.__onYesButtonClicked)
        self.cancelButton.clicked.connect(self.__onCancelButtonClicked)

    def resizeEvent(self, e):
        w, h = self.width(), self.height()
        self.yesButton.move(24, h-56)
        self.yesButton.resize((w-60)//2, self.yesButton.height())
        self.cancelButton.move(self.yesButton.geometry().right()+12, h-56)
        self.cancelButton.resize((w-60)//2, self.cancelButton.height())

    def __onCancelButtonClicked(self):
        self.cancelSignal.emit()
        self.deleteLater()

    def __onYesButtonClicked(self):
        self.yesSignal.emit()
        self.deleteLater()

    def __setQss(self):
        """ 设置层叠样式 """
        self.titleLabel.setObjectName("titleLabel")
        self.contentLabel.setObjectName("contentLabel")
        self.yesButton.setObjectName('yesButton')

        f = QFile(":/qss/dialog.qss")
        f.open(QFile.ReadOnly)
        self.setStyleSheet(str(f.readAll(), encoding='utf-8'))
        f.close()

        self.yesButton.adjustSize()
        self.cancelButton.adjustSize()
        self.contentLabel.adjustSize()
        self.titleLabel.adjustSize()

    def paintEvent(self, e):
        """ 绘制背景 """
        super().paintEvent(e)
        painter = QPainter(self)

        # 绘制文本区背景
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(Qt.white))
        painter.drawRect(0, 0, self.width(), self.height()-81)

        # 绘制分割线
        painter.setPen(QColor(229, 229, 229))
        painter.drawLine(0, self.height()-81,
                         self.width()-1, self.height()-81)

        # 绘制按钮背景
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(243, 243, 243))
        painter.drawRect(0, self.height()-80, self.width(), 80)
