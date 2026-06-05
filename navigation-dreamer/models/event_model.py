# models/event_model.py

import torch
import torch.nn as nn

from models.common import init_weights


class EventModel(nn.Module):
    """
    事件预测模型。

    输入：
        RSSM feature: (B, feature_dim) 或 (B, T, feature_dim)

    输出：
        logits: (B, 3) 或 (B, T, 3)

    三个维度分别表示：
        0: success
        1: collision
        2: stuck
    """

    def __init__(
        self,
        feature_dim: int = 288,
        hidden_dim: int = 256,
        num_events: int = 3,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_events = num_events

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),

            nn.Linear(hidden_dim, num_events),
        )

        self.apply(init_weights)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feat 最后一维应该是 {self.feature_dim}，但得到 {feat.shape}"
            )

        return self.net(feat)

    def event_prob(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(feat))