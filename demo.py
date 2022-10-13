# coding:utf-8
from algorithm.net import VOCDataset
from algorithm.utils.detection_utils import image_detect

# 模型文件和图片路径
model_path = 'model/Yolo_120.pth'
image_path = 'algorithm/resource/image/三上老师.jpg'

# 检测目标
anchors = [
    [[100, 146], [147, 203], [208, 260]],
    [[26, 43], [44, 65], [65, 105]],
    [[4, 8], [8, 15], [15, 27]]
]
image = image_detect(model_path, image_path, VOCDataset.classes, anchors=anchors, conf_thresh=0.5)
image.show()