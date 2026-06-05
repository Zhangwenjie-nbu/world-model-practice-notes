# models/decoder.py

import torch
import torch.nn as nn

from models.common import init_weights


class VisualDecoder(nn.Module):
    """
    视觉解码器。

    功能：
    将 RSSM latent feature 解码为 RGB 图像。

    输入：
        feat:
            RSSM feature，形状可以是：
            1. (B, feature_dim)
            2. (B, T, feature_dim)

    输出：
        recon:
            重建图像，数值范围为 [0, 1]。
            1. 如果输入是 (B, feature_dim)，输出为 (B, 3, 64, 64)
            2. 如果输入是 (B, T, feature_dim)，输出为 (B, T, 3, 64, 64)
    """

    def __init__(
        self,
        feature_dim: int = 288,
        image_size: int = 64,
        out_channels: int = 3,
        hidden_channels: int = 256,
    ):
        super().__init__()

        if image_size != 64:
            raise ValueError(
                "当前第一版 VisualDecoder 默认只支持 image_size=64。"
                "后续可以扩展为自适应尺寸。"
            )

        self.feature_dim = feature_dim
        self.image_size = image_size
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_channels * 4 * 4),
            nn.LayerNorm(hidden_channels * 4 * 4),
            nn.ELU(),
        )

        self.deconv = nn.Sequential(
            # 输入: B x 256 x 4 x 4
            nn.ConvTranspose2d(hidden_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # 输出: B x 128 x 8 x 8

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # 输出: B x 64 x 16 x 16

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            # 输出: B x 32 x 32 x 32

            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
            # 输出: B x 3 x 64 x 64，范围 [0, 1]
        )

        self.apply(init_weights)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            feat:
                RSSM feature。
                支持：
                - (B, feature_dim)
                - (B, T, feature_dim)

        返回：
            recon:
                重建图像。
                - 输入 (B, feature_dim) 时，输出 (B, 3, 64, 64)
                - 输入 (B, T, feature_dim) 时，输出 (B, T, 3, 64, 64)
        """

        if feat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feat 最后一维应该是 feature_dim={self.feature_dim}，"
                f"但得到 {feat.shape}"
            )

        leading_shape = feat.shape[:-1]

        # 把所有前置维度展平，例如：
        # (B, T, F) -> (B*T, F)
        flat_feat = feat.reshape(-1, self.feature_dim)

        x = self.fc(flat_feat)
        x = x.reshape(-1, self.hidden_channels, 4, 4)

        recon = self.deconv(x)

        # 还原前置维度，例如：
        # (B*T, C, H, W) -> (B, T, C, H, W)
        recon = recon.reshape(
            *leading_shape,
            self.out_channels,
            self.image_size,
            self.image_size,
        )

        return recon