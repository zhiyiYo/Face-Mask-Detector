# coding:utf-8
import sys

from PyQt5.QtCore import QLocale, Qt, QTranslator
from PyQt5.QtWidgets import QApplication

from app.view.main_window import MainWindow

# 启动 DPI 缩放
QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)


app = QApplication(sys.argv)

# 国际化
translator = QTranslator()
translator.load(QLocale.system(), ":/i18n/Face-Mask-Detector_")
app.installTranslator(translator)

w = MainWindow()
w.show()
app.exec_()
