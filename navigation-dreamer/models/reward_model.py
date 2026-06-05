# models/reward_model.py

import torch
import torch.nn as nn

from models.common import init_weights


class RewardModel(nn.Module):
    """
    Reward Model。

    功能：
    根据 RSSM latent feature 预测当前状态对应的 reward。

    输入：
        feat:
            RSSM feature，形状可以是：
            1. (B, feature_dim)
            2. (B, T, feature_dim)

    输出：
        reward:
            预测 reward，形状为：
            1. (B, 1)
            2. (B, T, 1)

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
                支持形状：
                - (B, feature_dim)
                - (B, T, feature_dim)

        返回：
            reward:
                预测 reward。
                - 输入是 (B, feature_dim)，输出为 (B, 1)
                - 输入是 (B, T, feature_dim)，输出为 (B, T, 1)
        """

        if feat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feat 最后一维应该是 feature_dim={self.feature_dim}，"
                f"但得到 {feat.shape}"
            )

        reward = self.net(feat)

        return reward