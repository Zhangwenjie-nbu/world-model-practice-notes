# models/continue_model.py

import torch
import torch.nn as nn

from models.common import init_weights


class ContinueModel(nn.Module):
    """
    Continue / Done Model。

    输入：
        RSSM feature: (B, feature_dim) 或 (B, T, feature_dim)

    输出：
        continue logits:
            没有 sigmoid 的 logits。
            训练时使用 BCEWithLogitsLoss。
    """

    def __init__(
        self,
        feature_dim: int = 288,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.feature_dim = feature_dim

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
        if feat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feat 最后一维应该是 {self.feature_dim}，但得到 {feat.shape}"
            )

        return self.net(feat)

    def continue_prob(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(feat))