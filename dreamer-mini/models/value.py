import torch
import torch.nn as nn


class Value(nn.Module):
    """
    Dreamer 的价值网络（Critic）

    输入:
        latent feature: [B, feat_dim]

    输出:
        value: [B, 1]
    """

    def __init__(self, feat_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat):
        """
        输入:
            feat: [B, feat_dim]

        输出:
            value: [B, 1]
        """
        return self.net(feat)