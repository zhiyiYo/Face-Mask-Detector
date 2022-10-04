# coding:utf-8
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from net import EvalPipeline, VOCDataset

mpl.rc_file('resource/theme/matlab.mplstyle')


# load dataset
root = 'data/VOCtest_06-Nov-2007/VOCdevkit/VOC2007'
dataset = VOCDataset(root, 'test')

# list all models
model_dir = Path('model/2021-11-29_14-30-54')
model_paths = [i for i in model_dir.glob('Yolo_*')]
model_paths.sort(key=lambda i: int(i.stem.split("_")[1]))

# evaluate models in list
mAPs = []
iterations = []
for model_path in model_paths:
    iterations.append(int(model_path.stem[5:]))
    ep = EvalPipeline(model_path, dataset, conf_thresh=0.001)
    mAPs.append(ep.eval()*100)

# save data
with open('eval/mAPs.json', 'w', encoding='utf-8') as f:
    json.dump(mAPs, f)

# plot mAP curve
fig, ax = plt.subplots(1, 1, num='mAP 曲线')
ax.plot(iterations, mAPs)
ax.set(xlabel='iteration', ylabel='mAP', title='mAP curve')
plt.show()
