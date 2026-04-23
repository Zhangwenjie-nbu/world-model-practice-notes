import torch
import torch.nn as nn


class ObsHead(nn.Module):
    """
    用 latent feature 预测 observation
    """

    def __init__(self, feat_dim, obs_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, feat):
        """
        输入:
            feat: [B, T, feat_dim]

        输出:
            obs_pred: [B, T, obs_dim]
        """
        return self.net(feat)


class RewardHead(nn.Module):
    """
    用 latent feature 预测 reward
    """

    def __init__(self, feat_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat):
        """
        输入:
            feat: [B, T, feat_dim]

        输出:
            reward: [B, T, 1]
        """
        return self.net(feat)