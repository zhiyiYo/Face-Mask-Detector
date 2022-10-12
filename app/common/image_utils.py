# coding: utf-8
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
import numpy as np


def imageToQPixmap(image: Image.Image):
    """ 将图像转换为 `QPixmap`

    Parameters
    ----------
    image: `~PIL.Image` or `np.ndarray`
        RGB 图像
    """
    image = np.array(image)  # type:np.ndarray
    h, w, c = image.shape
    format = QImage.Format_RGB888 if c == 3 else QImage.Format_RGBA8888
    return QPixmap.fromImage(QImage(image.data, w, h, c * w, format))


def rgb565ToImage(pixels: list) -> QPixmap:
    """ 将 RGB565 图像转换为 RGB888 """
    image = []
    for i in range(0, len(pixels), 2):
        pixel = (pixels[i] << 8) | pixels[i+1]
        r = pixel >> 11
        g = (pixel >> 5) & 0x3f
        b = pixel & 0x1f
        r = r * 255.0 / 31.0
        g = g * 255.0 / 63.0
        b = b * 255.0 / 31.0
        image.append([r, g, b])

    image = np.array(image, dtype=np.uint8).reshape(
        (240, 320, 3)).transpose((1, 0, 2))
    return imageToQPixmap(Image.fromarray(image))
