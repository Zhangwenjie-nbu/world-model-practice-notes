# models/actor.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

from models.common import init_weights


class DiscreteActor(nn.Module):
    """
    离散动作 Actor / Policy 网络。

    功能：
    根据 RSSM latent feature 输出离散动作分布。

    输入：
        feat:
            RSSM feature，形状可以是：
            1. (B, feature_dim)
            2. (B, T, feature_dim)

    输出：
        dist:
            torch.distributions.Categorical 分布。
            可以使用：
            - dist.sample() 采样动作
            - dist.log_prob(action) 计算 log probability
            - dist.entropy() 计算策略熵
            - dist.probs 查看动作概率
    """

    def __init__(
        self,
        feature_dim: int = 288,
        hidden_dim: int = 256,
        num_actions: int = 3,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),

            nn.Linear(hidden_dim, num_actions),
        )

        self.apply(init_weights)

    def get_logits(self, feat: torch.Tensor) -> torch.Tensor:
        """
        根据 RSSM feature 输出动作 logits。

        参数：
            feat:
                RSSM feature。
                支持：
                - (B, feature_dim)
                - (B, T, feature_dim)

        返回：
            logits:
                动作 logits。
                - 输入 (B, feature_dim)，输出 (B, num_actions)
                - 输入 (B, T, feature_dim)，输出 (B, T, num_actions)
        """

        if feat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"feat 最后一维应该是 feature_dim={self.feature_dim}，"
                f"但得到 {feat.shape}"
            )

        logits = self.net(feat)

        return logits

    def forward(self, feat: torch.Tensor, temperature: float = 1.0) -> Categorical:
        """
        返回动作分布。

        参数：
            feat:
                RSSM feature。

            temperature:
                温度系数，用于调节动作分布的平滑程度。
                temperature 越小，分布越尖锐；
                temperature 越大，分布越平滑。

        返回：
            dist:
                Categorical 动作分布。
        """

        if temperature <= 0:
            raise ValueError(f"temperature 必须大于 0，但得到 {temperature}")

        logits = self.get_logits(feat)

        logits = logits / temperature

        dist = Categorical(logits=logits)

        return dist

    def sample_action(
        self,
        feat: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ):
        """
        根据当前 feature 选择动作。

        参数：
            feat:
                RSSM feature。

            deterministic:
                如果为 True，选择概率最大的动作；
                如果为 False，从动作分布中随机采样。

            temperature:
                动作分布温度。

        返回：
            action:
                动作索引。
                - 输入 (B, feature_dim)，输出 (B,)
                - 输入 (B, T, feature_dim)，输出 (B, T)

            log_prob:
                所选动作的 log probability。

            entropy:
                当前动作分布的 entropy。
        """

        dist = self.forward(feat, temperature=temperature)

        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy