import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Dreamer 的策略网络

    输入:
        latent feature: [B, feat_dim]

    输出:
        action: [B, action_dim]
        dist info（用于后续 loss）
    """

    def __init__(self, feat_dim, action_dim, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, feat):
        """
        输入:
            feat: [B, feat_dim]

        输出:
            action
            mean, std（用于计算 log_prob）
        """
        x = self.net(feat)

        mean = self.mean(x)
        log_std = self.log_std(x)

        # 限制 std 范围（非常重要）
        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std)

        # reparameterization
        eps = torch.randn_like(mean)
        action = mean + std * eps

        # tanh squash
        action = torch.tanh(action)

        return action, mean, std