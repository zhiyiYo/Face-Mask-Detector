# coding:utf-8
import random
from os import path
from typing import List, Tuple, Union

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.annotation_utils import (AnnotationReaderBase,
                                      VocAnnotationReader,
                                      YoloAnnotationReader)
from ..utils.augmentation_utils import Transformer
from ..utils.box_utils import corner_to_center_numpy


class DatasetBase(Dataset):
    """ 数据集基类 """

    classes = [
        'face', 'face_mask'
    ]

    def __init__(self, root: Union[str, List[str]], transformer: Transformer = None,
                 color_transformer: Transformer = None, use_mosaic=False, use_mixup=False,
                 image_size=416):
        """
        Parameters
        ----------
        root: str or List[str]
            数据集的根路径

        transformer: Transformer
            不使用马赛克增强时用的数据增强器

        color_transformer: Transformer
            使用马赛克增强时用的数据增强器

        use_mosaic: bool
            是否启用马赛克数据增强

        use_mixup: bool
            是否启用 mixup 数据增强，只在 `mosaic` 为 `True` 时起作用

        image_size: int
            数据集输出的图像大小
        """
        if isinstance(root, str):
            root = [root]

        self.root = root
        self.image_size = image_size
        self.use_mosaic = use_mosaic
        self.use_mixup = use_mixup

        self.n_classes = len(self.classes)
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}

        self.transformer = transformer    # 数据增强器
        self.color_transformer = color_transformer

        self.annotation_reader = AnnotationReaderBase()

        # 指定数据集中的所有图片和标签文件路径
        self.image_names = []
        self.image_paths = []
        self.annotation_paths = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """ 获取样本

        Parameters
        ----------
        index: int
            下标

        Returns
        -------
        image: Tensor of shape `(3, H, W)`
            增强后的图像数据

        target: `np.ndarray` of shape `(n_objects, 5)`
            标签数据，每一行格式为 `(cx, cy, w, h, class)`
        """
        # 50% 的概率进行马赛克数据增强
        if self.use_mosaic and np.random.randint(2):
            image, bbox, label = self.make_mosaic(index)

            # mixup
            if self.use_mixup and np.random.randint(2):
                index_ = np.random.randint(0, len(self))
                image_, bbox_, label_ = self.make_mosaic(index_)
                r = np.random.beta(8, 8)
                image = (image*r+image_*(1-r)).astype(np.uint8)
                bbox = np.vstack((bbox, bbox_))
                label = np.hstack((label, label_))

            # 图像增强
            if self.color_transformer:
                image, bbox, label = self.color_transformer.transform(
                    image, bbox, label)

        else:
            image, bbox, label = self.read_image_label(index)
            if self.transformer:
                image, bbox, label = self.transformer.transform(
                    image, bbox, label)

        image = image.astype(np.float32)
        image /= 255.0
        bbox = corner_to_center_numpy(bbox)
        target = np.hstack((bbox, label[:, np.newaxis]))

        return torch.from_numpy(image).permute(2, 0, 1), target

    def make_mosaic(self, index: int):
        """ 创建马赛克增强后的图片 """
        # 随机选择三张图
        indexes = list(range(len(self.image_paths)))
        choices = random.sample(indexes[:index]+indexes[index+1:], 3)
        choices.append(index)

        # 读入用于制作马赛克的四张图及其标签
        images, bboxes, labels = [], [], []
        for i in choices:
            image, bbox, label = self.read_image_label(i)
            images.append(image)
            bboxes.append(bbox)
            labels.append(label)

        # 创建马赛克图像并选取拼接点
        img_size = self.image_size
        mean = np.array([123, 117, 104])
        mosaic_img = np.ones((img_size*2, img_size*2, 3))*mean
        xc = int(random.uniform(img_size//2, 3*img_size//2))
        yc = int(random.uniform(img_size//2, 3*img_size//2))

        # 拼接图像
        for i, (image, bbox, label) in enumerate(zip(images, bboxes, labels)):
            # 保留/不保留比例缩放图像
            ih, iw, _ = image.shape
            s = np.random.choice(np.arange(50, 210, 10))/100
            if np.random.randint(2):
                r = img_size / max(ih, iw)
                if r != 1:
                    image = cv.resize(image, (int(iw*r*s), int(ih*r*s)))
            else:
                image = cv.resize(image, (int(img_size*s), int(img_size*s)))

            # 将图像粘贴到拼接点的左上角、右上角、左下角和右下角
            h, w, _ = image.shape
            if i == 0:
                # 粘贴部分的左上角和右下角在马赛克图像中的坐标
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                # 粘贴部分的左上角和右下角在原始图像中的坐标
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(
                    yc - h, 0), min(xc + w, img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(
                    xc - w, 0), yc, xc, min(img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(
                    xc + w, img_size * 2), min(img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]

            # 将边界框反归一化并平移坐标
            dx = x1a - x1b
            dy = y1a - y1b
            bbox[:, [0, 2]] = bbox[:, [0, 2]]*w+dx
            bbox[:, [1, 3]] = bbox[:, [1, 3]]*h+dy

        # 处理超出马赛克图像坐标系的边界框
        bbox = np.clip(np.vstack(bboxes), 0, 2*img_size)
        label = np.hstack(labels)

        # 移除过小的边界框
        bbox_w = bbox[:, 2] - bbox[:, 0]
        bbox_h = bbox[:, 3] - bbox[:, 1]
        mask = np.logical_and(bbox_w > 3, bbox_h > 3)
        bbox, label = bbox[mask], label[mask]
        if len(bbox) == 0:
            bbox = np.zeros((1, 4))
            label = np.array([0])

        # 归一化边界框
        bbox /= mosaic_img.shape[0]

        return mosaic_img, bbox, label

    def read_image_label(self, index: int):
        """ 读取图片和标签数据

        Parameters
        ----------
        index: int
            读取的样本索引

        Returns
        -------
        image: `~np.ndarray` of shape `(H, W, 3)`
                RGB 图像

        bbox: `~np.ndarray` of shape `(n_objects, 4)`
            已经归一化的边界框

        label: `~np.ndarray` of shape `(n_objects, )`
            类别标签
        """
        image = cv.cvtColor(
            cv.imread(self.image_paths[index]), cv.COLOR_BGR2RGB)
        target = np.array(self.annotation_reader.read(
            self.annotation_paths[index]))
        bbox, label = target[:, :4], target[:, 4]
        return image, bbox, label


class VOCDataset(DatasetBase):
    """ VOC 数据集 """

    def __init__(self, root: Union[str, List[str]], image_set: Union[str, List[str]],
                 transformer: Transformer = None, color_transformer: Transformer = None, keep_difficult=False,
                 use_mosaic=False, use_mixup=False, image_size=416):
        """
        Parameters
        ----------
        root: str or List[str]
            数据集的根路径，下面必须有 `Annotations`、`ImageSets` 和 `JPEGImages` 文件夹

        image_set: str or List[str]
            数据集的种类，可以是 `train`、`val`、`trainval` 或者 `test`

        transformer: Transformer
            不使用马赛克增强时用的数据增强器

        color_transformer: Transformer
            使用马赛克增强时用的数据增强器

        keep_difficulty: bool
            是否保留 difficult 为 1 的样本

        use_mosaic: bool
            是否启用马赛克数据增强

        use_mixup: bool
            是否启用 mixup 数据增强，只在 `mosaic` 为 `True` 时起作用

        image_size: int
            数据集输出的图像大小
        """
        super().__init__(root, transformer, color_transformer,
                         use_mosaic, use_mixup, image_size)
        if isinstance(image_set, str):
            image_set = [image_set]
        if len(self.root) != len(image_set):
            raise ValueError("`root` 和 `image_set` 的个数必须相同")

        self.image_set = image_set

        self.annotation_reader = VocAnnotationReader(
            self.class_to_index, keep_difficult)

        for root, image_set in zip(self.root, self.image_set):
            with open(path.join(root, f'ImageSets/Main/{image_set}.txt')) as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    self.image_names.append(line)
                    self.image_paths.append(
                        path.join(root, f'JPEGImages/{line}.jpg'))
                    self.annotation_paths.append(
                        path.join(root, f'Annotations/{line}.xml'))


class YoloDataset(DatasetBase):
    """ Yolo 数据集 """

    def __init__(self, root: Union[str, List[str]], image_set: Union[str, List[str]],
                 transformer: Transformer = None, color_transformer: Transformer = None,
                 use_mosaic=False, use_mixup=False, image_size=416):
        """
        Parameters
        ----------
        root: str or List[str]
            数据集的根路径，下面必须有 `labels` 和 `images` 文件夹，以及 `${image_set}.txt`

        transformer: Transformer
            不使用马赛克增强时用的数据增强器

        color_transformer: Transformer
            使用马赛克增强时用的数据增强器

        use_mosaic: bool
            是否启用马赛克数据增强

        use_mixup: bool
            是否启用 mixup 数据增强，只在 `mosaic` 为 `True` 时起作用

        image_size: int
            数据集输出的图像大小
        """
        super().__init__(root, transformer, color_transformer,
                         use_mosaic, use_mixup, image_size)
        if isinstance(image_set, str):
            image_set = [image_set]
        if len(self.root) != len(image_set):
            raise ValueError("`root` 和 `image_set` 的个数必须相同")

        self.image_set = image_set
        self.annotation_reader = YoloAnnotationReader()

        for root, image_set in zip(self.root, self.image_set):
            with open(path.join(root, f'{image_set}.txt')) as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    self.image_names.append(line)
                    self.image_paths.append(
                        path.join(root, f'images/{line}.jpg'))
                    self.annotation_paths.append(
                        path.join(root, f'labels/{line}.txt'))


def collate_fn(batch: List[Tuple[torch.Tensor, np.ndarray]]):
    """ 整理 dataloader 取出的数据

    Parameters
    ----------
    batch: list of shape `(N, 2)`
        一批数据，列表中的每一个元组包括两个元素：
        * image: Tensor of shape `(3, H, W)`
        * target: `~np.ndarray` of shape `(n_objects, 5)`

    Returns
    -------
    image: Tensor of shape `(N, 3, H, W)`
        图像

    target: List[Tensor]
        标签
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img.to(torch.float32))
        targets.append(torch.Tensor(target))

    return torch.stack(images, 0), targets


def make_data_loader(dataset: DatasetBase, batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
