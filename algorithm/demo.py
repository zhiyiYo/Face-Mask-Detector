# coding:utf-8
from net import VOCDataset
from utils.detection_utils import image_detect

# 模型文件和图片路径
model_path = 'model/Yolo_140.pth'
image_path = 'data/images/WIN_20220803_19_12_45_Pro_flip.jpg'

# 检测目标
anchors = [
    [[216, 332], [342, 234], [330, 347]],
    [[249, 105], [128, 279], [300, 145]],
    [[58,  58], [179,  87], [90, 192]],
]
image = image_detect(model_path, image_path, VOCDataset.classes, anchors=anchors,conf_thresh=0.5)
image.show()