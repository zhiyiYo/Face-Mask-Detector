# coding:utf-8
from torch.optim import Optimizer
from math import pi, cos


class WarmUpCosLRSchedule:
    """ 热启动余弦学习率规划器 """

    def __init__(self, optimizer: Optimizer, lr: float, min_lr: float, total_epoch: int, warm_up_ratio=0.05, no_aug_ratio=0.05, warm_up_factor=1/3):
        """
        Parameters
        ----------
        optimizer: Optimizer
            优化器

        lr: float
            初始学习率

        min_lr: float
            收尾阶段的学习率

        total_iters: int
            总共迭代的世代数

        warm_up_ratio: int
            热启动迭代比例

        no_aug_ratio: float
            没有马赛克数据增强的迭代比例

        warm_up_factor: float
            第一次迭代时热启动学习率和初始学习率的比值
        """
        self.lr = lr
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.warm_up_factor = warm_up_factor
        self.total_epoch = total_epoch
        self.warm_up_epoch = int(warm_up_ratio*total_epoch)
        self.no_aug_epoch = int(no_aug_ratio*total_epoch)

    def step(self, epoch: int):
        """ 调整优化器的学习率 """
        if epoch < self.warm_up_epoch:
            delta = (1 - self.warm_up_factor) * epoch / self.warm_up_epoch
            lr = (self.warm_up_factor + delta) * self.lr
        elif epoch >= self.total_epoch-self.no_aug_epoch:
            lr = self.min_lr
        else:
            cos_iters = self.total_epoch - self.warm_up_factor - self.no_aug_epoch
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1 + cos(pi * (epoch - self.warm_up_epoch) / cos_iters)
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def set_lr(self, lr, min_lr):
        """ 设置学习率 """
        self.lr = lr
        self.min_lr = min_lr


def get_lr(optimizer):
    """ 获取当前学习率 """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def determin_lr(lr, batch_size):
    """ 根据批大小计算学习率 """
    bs = 64
    lr_max = 5e-2
    lr_min = 5e-4
    lr_fit = min(max(batch_size/bs*lr, lr_min), lr_max)
    lr_min_fit = min(max(batch_size/bs*(lr/100), lr_min/100), lr_max/100)
    return lr_fit, lr_min_fit
