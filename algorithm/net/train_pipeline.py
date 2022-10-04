# coding:utf-8
import time
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
from torch import cuda
from torch.backends import cudnn
from tqdm import tqdm
from utils.log_utils import LossLogger, Logger
from utils.datetime_utils import time_delta
from utils.lr_schedule_utils import WarmUpCosLRSchedule, determin_lr, get_lr
from utils.optimizer_utils import make_optimizer

from .dataset import VOCDataset, make_data_loader
from .loss import YoloLoss
from .yolo import Yolo


def exception_handler(train_func):
    """ 处理训练过程中发生的异常并保存模型 """
    def wrapper(train_pipeline, *args, **kwargs):
        try:
            return train_func(train_pipeline, *args, **kwargs)
        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):
                Logger("error").error(f"{e.__class__.__name__}: {e}", True)

            train_pipeline.save()

            # 清空 GPU 缓存
            cuda.empty_cache()

            exit()

    return wrapper


class TrainPipeline:
    """ 训练模型流水线 """

    def __init__(self, n_classes: int, image_size: int, anchors: list, dataset: VOCDataset, darknet_path: str = None,
                 yolo_path: str = None, lr=0.01, momentum=0.9, weight_decay=4e-5, warm_up_ratio=0.02, freeze=True,
                 batch_size=4, freeze_batch_size=8, num_workers=4, freeze_epoch=20, start_epoch=0, max_epoch=60,
                 save_frequency=5, use_gpu=True, save_dir='model', log_file: str = None, log_dir='log'):
        """
        Parameters
        ----------
        n_classes: int
            类别数

        image_size: int
            输入 Yolo 神经网络的图片大小

        anchors: list of shape `(3, n_anchors, 2)`
            输入神经网络的图片尺寸为 416 时的先验框尺寸

        dataset: Dataset
            训练数据集

        darknet_path: str
            预训练的 darknet53 模型文件路径

        yolo_path: Union[str, None]
            Yolo 模型文件路径，有以下两种选择:
            * 如果不为 `None`，将使用模型文件中的参数初始化 `Yolo`
            * 如果为 `None`，将随机初始化 darknet53 之后的各层参数

        lr: float
            学习率

        momentum: float
            冲量

        weight_decay: float
            权重衰减

        warm_up_ratio: float
            暖启动的世代占全部世代的比例

        freeze: bool
            是否使用冻结训练

        batch_size: int
            非冻结训练过程训练集的批大小

        freeze_batch_size: int
            冻结训练过程中的批大小

        num_workers: int
            加载数据的线程数

        freeze_epoch: int
            冻结训练的世代数

        start_epoch: int
            Yolo 模型文件包含的参数是训练了多少个世代的结果

        max_epoch: int
            最多迭代多少个世代

        save_frequency: int
            迭代多少个世代保存一次模型

        use_gpu: bool
            是否使用 GPU 加速训练

        save_dir: str
            保存 SSD 模型的文件夹

        log_file: str
            训练损失数据历史记录文件，要求是 json 文件

        save_dir: str
            训练损失数据保存的文件夹
        """
        self.dataset = dataset
        self.save_dir = Path(save_dir)
        self.use_gpu = use_gpu
        self.save_frequency = save_frequency
        self.freeze_batch_size = freeze_batch_size
        self.batch_size = batch_size

        self.lr = lr
        self.freeze = freeze
        self.max_epoch = max_epoch
        self.start_epoch = start_epoch
        self.current_epoch = start_epoch
        self.free_epoch = freeze_epoch

        if use_gpu and cuda.is_available():
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # 创建模型
        self.model = Yolo(n_classes, image_size, anchors).to(self.device)
        if yolo_path:
            self.model.load(yolo_path)
            print('🧪 成功载入 Yolo 模型：' + yolo_path)
        elif darknet_path:
            self.model.backbone.load(darknet_path)
            print('🧪 成功载入 Darknet53 模型：' + darknet_path)
        else:
            raise ValueError("必须指定预训练的 Darknet53 模型文件路径")

        self.model.backbone.set_freezed(freeze)

        # 创建优化器和损失函数
        bs = freeze_batch_size if freeze else batch_size
        lr_fit, lr_min_fit = determin_lr(lr, bs)
        self.criterion = YoloLoss(anchors, n_classes, image_size)
        self.optimizer = make_optimizer(
            self.model, lr_fit, momentum, weight_decay)
        self.lr_schedule = WarmUpCosLRSchedule(
            self.optimizer, lr_fit, lr_min_fit, max_epoch, warm_up_ratio)

        # 数据集加载器
        self.num_worksers = num_workers
        self.n_batches = len(self.dataset)//bs
        self.data_loader = make_data_loader(self.dataset, bs, num_workers)

        # 训练损失记录器
        self.logger = LossLogger(log_file, log_dir)

    def save(self):
        """ 保存模型和训练损失数据 """
        self.pbar.close()
        self.save_dir.mkdir(exist_ok=True, parents=True)

        # 保存模型
        self.model.eval()
        path = self.save_dir/f'Yolo_{self.current_epoch+1}.pth'
        torch.save(self.model.state_dict(), path)

        # 保存训练损失数据
        self.logger.save(f'train_losses_{self.current_epoch+1}')

        print(f'\n🎉 已将当前模型保存到 {path.absolute()}\n')

    @exception_handler
    def train(self):
        """ 训练模型 """
        t = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        self.save_dir = self.save_dir/t
        self.logger.save_dir = self.logger.save_dir/t

        bar_format = '{desc}{n_fmt:>4s}/{total_fmt:<4s}|{bar}|{postfix}'
        print('🚀 开始训练！')

        is_unfreezed = False
        for e in range(self.start_epoch, self.max_epoch):
            self.current_epoch = e

            # 解冻训练
            if self.freeze and e >= self.free_epoch and not is_unfreezed:
                print('\n🧊 开始解冻训练！\n')
                is_unfreezed = True
                self.lr_schedule.set_lr(*determin_lr(self.lr, self.batch_size))
                self.data_loader = make_data_loader(
                    self.dataset, self.batch_size, self.num_worksers)
                self.n_batches = len(self.dataset)//self.batch_size
                self.model.backbone.set_freezed(False)

            self.model.train()

            # 创建进度条
            self.pbar = tqdm(total=self.n_batches, bar_format=bar_format)
            self.pbar.set_description(f"\33[36m💫 Epoch {(e+1):5d}/{self.max_epoch}")
            start_time = datetime.now()

            loss_value = 0
            for iter, (images, targets) in enumerate(self.data_loader, 1):
                # 预测边界框、置信度和条件类别概率
                preds = self.model(images.to(self.device))

                # 误差反向传播
                self.optimizer.zero_grad()
                loss = 0
                for i, pred in enumerate(preds):
                    loss += self.criterion(i, pred, targets)

                loss.backward()
                self.optimizer.step()

                # 记录误差
                loss_value += loss.item()

                # 更新进度条
                cost_time = time_delta(start_time)
                self.pbar.set_postfix_str(
                    f'loss: {loss_value/iter:.4f}, lr: {get_lr(self.optimizer):.5f}, time: {cost_time}\33[0m')
                self.pbar.update()

            # 将误差写入日志
            self.logger.record(loss_value/iter)

            # 关闭进度条
            self.pbar.close()

            # 学习率退火
            self.lr_schedule.step(e)

            # 关闭马赛克数据增强
            if e == self.max_epoch - self.lr_schedule.no_aug_epoch:
                self.dataset.use_mosaic = False

            # 定期保存模型
            if e > self.start_epoch and (e+1-self.start_epoch) % self.save_frequency == 0:
                self.save()

        self.save()
