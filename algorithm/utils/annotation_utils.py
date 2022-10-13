# coding: utf-8
from pathlib import Path
from typing import Dict, Union
from xml.etree import ElementTree as ET


class AnnotationReaderBase:
    """ 标签读取器基类 """

    def read(self, file_path: Union[str, Path]):
        """ 解析标签文件

        Parameters
        ----------
        file_path: str
            文件路径

        Returns
        -------
        target: List[list] of shape `(n_objects, 5)`
            标签列表，每个标签的前四个元素为归一化后的边界框，最后一个标签为编码后的类别，
            e.g. `[[xmin, ymin, xmax, ymax, class], ...]`
        """
        raise NotImplementedError


class VocAnnotationReader(AnnotationReaderBase):
    """ xml 格式的标注转换器 """

    def __init__(self, class_to_index: Dict[str, int], keep_difficult=False):
        """
        Parameters
        ----------
        class_to_index: Dict[str, int]
            类别:编码 字典

        keep_difficulty: bool
            是否保留 difficult 为 1 的样本
        """
        self.class_to_index = class_to_index
        self.keep_difficult = keep_difficult

    def read(self, file_path: str):
        root = ET.parse(file_path).getroot()

        # 图像的尺寸
        img_size = root.find('size')
        w = int(img_size.find('width').text)
        h = int(img_size.find('height').text)

        # 提取所有的标签
        target = []
        for obj in root.iter('object'):
            # 样本是否难以预测
            difficult = int(obj.find('difficult').text)
            if not self.keep_difficult and difficult:
                continue

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            # 归一化方框位置
            points = ['xmin', 'ymin', 'xmax', 'ymax']
            data = []
            for i, pt in enumerate(points):
                pt = int(bbox.find(pt).text) - 1
                pt = pt/w if i % 2 == 0 else pt/h
                data.append(pt)

            # 检查数据是否合法
            if data[0] >= data[2] or data[1] >= data[3]:
                p = [int(bbox.find(pt).text) for pt in points]
                raise ValueError(f"{file_path} 存在脏数据：object={name}, bbox={p}")

            data.append(self.class_to_index[name])
            target.append(data)

        return target


class YoloAnnotationReader(AnnotationReaderBase):
    """ Yolo 格式数据集读取器 """

    def read(self, file_path: str):
        target = []

        # 读取所有边界框
        with open(file_path) as f:
            lines = f.readlines()

        for line in lines:
            if not line.strip():
                continue

            c, cx, cy, w, h = line.strip().split()
            cx, cy, w, h = [float(i) for i in [cx, cy, w, h]]
            xmin = cx - w/2
            xmax = cx + w/2
            ymin = cy - h/2
            ymax = cy + h/2

            # 检查数据合法性
            if xmin >= xmax or ymin >= ymax:
                raise ValueError(
                    f"{file_path} 存在脏数据：object={c}, bbox={[cx, cy, w, h]}")

            target.append([xmin, ymin, xmax, ymax, int(c)])

        return target
