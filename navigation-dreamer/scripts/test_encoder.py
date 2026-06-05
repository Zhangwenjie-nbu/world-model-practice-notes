# scripts/test_encoder.py

import sys
from pathlib import Path

import numpy as np
import torch

# 让脚本可以从项目根目录导入模块
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from envs.grid_nav_env import GridNavEnv
from models.encoder import VisualEncoder
from models.common import count_parameters


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    将环境输出的 numpy 图像转换为 PyTorch Tensor。

    环境输出：
        obs.shape = (H, W, C)
        obs.dtype = uint8
        obs range = [0, 255]

    模型输入：
        tensor.shape = (1, C, H, W)
        tensor.dtype = float32
        tensor range = [0, 1]
    """

    assert obs.ndim == 3
    assert obs.shape[2] == 3
    assert obs.dtype == np.uint8

    tensor = torch.from_numpy(obs).float() / 255.0

    # HWC -> CHW
    tensor = tensor.permute(2, 0, 1)

    # CHW -> BCHW
    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"当前设备: {device}")
    print("=" * 60)

    env = GridNavEnv(
        map_width=10,
        map_height=10,
        image_size=64,
        max_steps=50,
        random_reset=True,
        seed=42,
    )

    obs = env.reset()

    print("[环境输出]")
    print(f"obs shape: {obs.shape}")
    print(f"obs dtype: {obs.dtype}")
    print(f"obs min/max: {obs.min()} / {obs.max()}")

    obs_tensor = obs_to_tensor(obs, device)

    print("\n[转换为模型输入]")
    print(f"obs_tensor shape: {obs_tensor.shape}")
    print(f"obs_tensor dtype: {obs_tensor.dtype}")
    print(f"obs_tensor min/max: {obs_tensor.min().item():.3f} / {obs_tensor.max().item():.3f}")

    encoder = VisualEncoder(
        image_size=64,
        in_channels=3,
        embedding_dim=256,
    ).to(device)

    encoder.eval()

    print("\n[Encoder 信息]")
    print(encoder)
    print(f"可训练参数量: {count_parameters(encoder):,}")

    with torch.no_grad():
        embedding = encoder(obs_tensor)

    print("\n[Encoder 输出]")
    print(f"embedding shape: {embedding.shape}")
    print(f"embedding dtype: {embedding.dtype}")
    print(f"embedding mean: {embedding.mean().item():.6f}")
    print(f"embedding std: {embedding.std().item():.6f}")

    assert embedding.shape == (1, 256)

    print("\n测试通过：VisualEncoder 可以正常处理环境图像。")


if __name__ == "__main__":
    main()