# models/encoder.py

import torch
import torch.nn as nn

from models.common import init_weights


class VisualEncoder(nn.Module):
    """
    视觉编码器。

    功能：
    将原始 RGB 图像观测编码为低维 embedding。

    输入：
        obs: torch.Tensor
            形状为 (B, 3, 64, 64)
            数值范围建议为 [0, 1]

    输出：
        embedding: torch.Tensor
            形状为 (B, embedding_dim)
    """

    def __init__(self, image_size: int = 64, in_channels: int = 3, embedding_dim: int = 256):
        super().__init__()

        if image_size != 64:
            raise ValueError(
                "当前第一版 VisualEncoder 默认只支持 image_size=64。"
                "后续可以扩展为自适应尺寸。"
            )

        self.image_size = image_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.cnn = nn.Sequential(
            # 输入: B x 3 x 64 x 64
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 输出: B x 32 x 32 x 32

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 输出: B x 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 输出: B x 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 输出: B x 256 x 4 x 4
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.apply(init_weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            obs:
                图像张量，形状为 (B, 3, 64, 64)，数值范围建议为 [0, 1]。

        返回：
            embedding:
                图像 embedding，形状为 (B, embedding_dim)。
        """

        if obs.ndim != 4:
            raise ValueError(f"obs 应该是 4 维张量 (B, C, H, W)，但得到 {obs.shape}")

        if obs.shape[1] != self.in_channels:
            raise ValueError(
                f"obs channel 应该是 {self.in_channels}，但得到 {obs.shape[1]}"
            )

        x = self.cnn(obs)
        embedding = self.fc(x)

        return embedding