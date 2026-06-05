# models/common.py

import torch
import torch.nn as nn


def init_weights(module: nn.Module):
    """
    通用神经网络参数初始化函数。

    当前策略：
    1. 对 Conv2d、ConvTranspose2d 和 Linear 使用 Xavier 初始化；
    2. bias 初始化为 0。
    """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> int:
    """
    统计模型中可训练参数数量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)