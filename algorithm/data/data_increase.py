# coding:utf-8
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2 as cv
from imgaug import augmenters as iaa
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage
import numpy as np

from net import YoloDataset
from utils.annotation_utils import YoloAnnotationReader
from utils.box_utils import corner_to_center_numpy


def get_all_image_names(root):
    image_names = []
    for image_path in Path(root+"/images").glob("*.jpg"):
        image_names.append(image_path.stem+"\n")
        anno_path=Path(root+"/labels")/(image_path.stem+".txt")
        if not anno_path.exists():
            print(anno_path, "不存在")

    with open(root+"/all.txt", "w") as f:
        f.writelines(image_names)

    return image_names


root = "data"
get_all_image_names(root)


dataset = YoloDataset(root, 'all')
annotation_reader = YoloAnnotationReader()


def rotate(image_path: str, anno_path: str):
    """ 图像顺时针旋转 90 度 """
    # 读入图像
    image = cv.imread(image_path)
    h, w, c = image.shape

    rot90 = iaa.Rot90(keep_size=False)
    data = np.array(annotation_reader.read(anno_path))
    labels = data[:, -1]
    boxes = []
    for box in data[:, :4]:
        box[[0, 2]] *= w
        box[[1, 3]] *= h
        boxes.append(BoundingBox(*box.tolist()))

    boxes = BoundingBoxesOnImage(boxes, image.shape)

    # 上下翻转图像和边界框
    image_, boxes = rot90(image=image, bounding_boxes=boxes)
    boxes = corner_to_center_numpy(boxes.to_xyxy_array(int))
    boxes[:, [0, 2]] /= h
    boxes[:, [1, 3]] /= w
    return image_, boxes, labels


def flip(image_path: str, anno_path: str):
    """ 图像上下翻转 """
    # 读入图像
    image = cv.imread(image_path)
    h, w, c = image.shape

    rot90 = iaa.Flipud()
    data = np.array(annotation_reader.read(anno_path))
    labels = data[:, -1]
    boxes = []
    for box in data[:, :4]:
        box[[0, 2]] *= w
        box[[1, 3]] *= h
        boxes.append(BoundingBox(*box.tolist()))

    boxes = BoundingBoxesOnImage(boxes, image.shape)

    # 上下翻转图像和边界框
    image_, boxes = rot90(image=image, bounding_boxes=boxes)
    boxes = corner_to_center_numpy(boxes.to_xyxy_array(int))
    boxes[:, [0, 2]] /= w
    boxes[:, [1, 3]] /= h
    return image_, boxes, labels


def save(image, boxes, labels, old_image_path, old_anno_path, suffix: str):
    """ 保存图像和标签 """
    old_image_path = Path(old_image_path)
    old_anno_path = Path(old_anno_path)
    image_path = old_image_path.with_name(
        f"{old_image_path.stem}_{suffix}.jpg")
    anno_path = old_anno_path.with_name(f"{old_anno_path.stem}_{suffix}.txt")

    lines = ""
    for box, c in zip(boxes, labels):
        lines += f'{int(c)} {box[0]} {box[1]} {box[2]} {box[3]}\n'

    anno_path.write_text(lines, encoding='utf-8')
    cv.imwrite(str(image_path), image)



for i, (image_path, anno_path) in enumerate(zip(dataset.image_paths, dataset.annotation_paths)):
    print(f'\r旋转图像中，当前进度: {i}/{len(dataset)}', end='')
    image, boxes, labels = rotate(image_path, anno_path)
    save(image, boxes, labels, image_path, anno_path, "rot90")

print('\n')

# 更新数据集
get_all_image_names(root)
dataset = YoloDataset('data', 'all')

for i, (image_path, anno_path) in enumerate(zip(dataset.image_paths, dataset.annotation_paths)):
    print(f'\r上下翻转图像中，当前进度: {i}/{len(dataset)}', end='')
    image, boxes, labels = flip(image_path, anno_path)
    save(image, boxes, labels, image_path, anno_path, "flip")

get_all_image_names(root)