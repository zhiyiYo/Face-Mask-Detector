# coding:utf-8
from net import TrainPipeline, VOCDataset
from utils.augmentation_utils import YoloAugmentation, ColorAugmentation


# train config
config = {
    "n_classes": len(VOCDataset.classes),
    "image_size": 416,
    "anchors": [
        [[100, 146], [147, 203], [208, 260]],
        [[26, 43], [44, 65], [65, 105]],
        [[4, 8], [8, 15], [15, 27]]
    ],
    "darknet_path": "model/CSPdarknet53.pth",
    "lr": 1e-2,
    "batch_size": 4,
    "freeze_batch_size": 8,
    "freeze": True,
    "freeze_epoch": 50,
    "max_epoch": 160,
    "start_epoch": 0,
    "num_workers": 4
}

# load dataset
root = 'data'
dataset = VOCDataset(
    root,
    'trainval',
    transformer=YoloAugmentation(config['image_size']),
    color_transformer=ColorAugmentation(config['image_size']),
    use_mosaic=False,
    use_mixup=True,
    image_size=config["image_size"]
)

if __name__ == '__main__':
    train_pipeline = TrainPipeline(dataset=dataset, **config)
    train_pipeline.train()
