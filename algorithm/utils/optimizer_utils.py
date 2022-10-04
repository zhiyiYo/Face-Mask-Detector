# coding: utf-8
from torch import optim, nn


def make_optimizer(model: nn.Module, lr, momentum=0.9, weight_decay=5e-4):
    """ 创建优化器 """
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    optimizer = optim.SGD(pg0, lr, momentum=momentum, nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})
    return optimizer