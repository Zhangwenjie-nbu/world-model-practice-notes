# models/visual_actor.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.common import init_weights


class VisualDiscreteActor(nn.Module):
    """
    直接从图像预测动作的 Actor。

    只用于 sanity check：
        如果这个模型能 BC 成功，而 RSSM Actor 不行，
        说明问题在 RSSM latent。
    """

    def __init__(self, num_actions: int = 4):
        super().__init__()

        self.num_actions = num_actions

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, num_actions),
        )

        self.apply(init_weights)

    def get_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs:
            (B, C, H, W) 或 (B, T, C, H, W)
        """
        if obs.ndim == 4:
            x = self.cnn(obs)
            return self.head(x)

        if obs.ndim == 5:
            b, t, c, h, w = obs.shape
            flat = obs.reshape(b * t, c, h, w)
            logits = self.get_logits(flat)
            return logits.reshape(b, t, self.num_actions)

        raise ValueError(f"Unsupported obs shape: {obs.shape}")

    def forward(self, obs: torch.Tensor):
        logits = self.get_logits(obs)
        return Categorical(logits=logits)

    def sample_action(self, obs: torch.Tensor, deterministic: bool = True):
        dist = self.forward(obs)

        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy