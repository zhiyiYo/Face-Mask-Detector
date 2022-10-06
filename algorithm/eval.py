# coding:utf-8
from net import EvalPipeline, VOCDataset

# load dataset
root = 'data/FaceMaskDataset/val'
dataset = VOCDataset(root, 'all')

anchors = [
    [[100, 146], [147, 203], [208, 260]],
    [[26, 43], [44, 65], [65, 105]],
    [[4, 8], [8, 15], [15, 27]]
]
model_path = 'model/Yolo_140.pth'
eval_pipeline = EvalPipeline(model_path, dataset, anchors=anchors, conf_thresh=0.001)
eval_pipeline.eval()