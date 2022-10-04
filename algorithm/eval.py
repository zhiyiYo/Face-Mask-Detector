# coding:utf-8
from net import EvalPipeline, VOCDataset

# load dataset
root = 'data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007'
dataset = VOCDataset(root, 'test')

model_path = 'model/2022-10-02_16-52-03/Yolo_50.pth'
eval_pipeline = EvalPipeline(model_path, dataset, conf_thresh=0.001)
eval_pipeline.eval()