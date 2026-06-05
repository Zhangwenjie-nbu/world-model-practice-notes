# models/critic.py

import torch
import torch.nn as nn

from models.common import init_weights


class Critic(nn.Module):
    """
    Critic / Value 网络。

    功能：
    根据 RSSM latent feature 估计当前状态的长期价值。

    输入：
        feat:
            RSSM feature，形状可以是：
            1. (B, feature_dim)
            2. (B, T, feature_dim)

    输出：
        value:
            状态价值估计。
            1. 输入为 (B, feature_dim)，输出为 (B, 1)
            2. 输入为 (B, T, feature_dim)，输出为 (B, T, 1)

    当前项目中：
        feature_dim = deter_dim + stoch_dim = 256 + 32 = 288
    """

    def __init__(
        self,
        feature_dim: int = 288,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),

            nn.Linear(hidden_dim, 1),
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
            value:
                value estimate。
                - 输入 (B, feature_dim)，输出 (B, 1)
                - 输入 (B, T, feature_dim)，输出 (B, T, 1)
        """

        if feat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feat 最后一维应该是 feature_dim={self.feature_dim}，"
                f"但得到 {feat.shape}"
            )

        value = self.net(feat)

        return value