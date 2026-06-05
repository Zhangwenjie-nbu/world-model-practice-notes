# tools/seed.py

import random
import numpy as np
import torch


def set_global_seed(seed: int):
    """
    设置 Python、NumPy、PyTorch 的随机种子，增强实验可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)