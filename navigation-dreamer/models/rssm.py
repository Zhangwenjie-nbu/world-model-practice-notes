# models/rssm.py

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import init_weights


@dataclass
class RSSMState:
    """
    RSSM 状态。

    deter:
        deterministic hidden state，形状为 (..., deter_dim)

    stoch:
        stochastic latent state，形状为 (..., stoch_dim)

    mean:
        stochastic state 对应的高斯均值，形状为 (..., stoch_dim)

    std:
        stochastic state 对应的高斯标准差，形状为 (..., stoch_dim)
    """
    deter: torch.Tensor
    stoch: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


class RSSM(nn.Module):
    """
    Action-conditioned RSSM。

    功能：
    1. 根据上一时刻 state 和上一动作，预测当前 prior state；
    2. 根据当前 embedding 修正得到 posterior state；
    3. 支持 observation rollout；
    4. 支持 imagination rollout。

    约定：
    - embedding e_t 来自当前观测 obs_t；
    - action a_{t-1} 是进入当前状态之前执行的动作；
    - observe() 中 actions[:, t] 表示 obs_t 对应的 previous action。
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_actions: int = 3,
        action_embed_dim: int = 32,
        deter_dim: int = 256,
        stoch_dim: int = 32,
        hidden_dim: int = 256,
        min_std: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std

        # 动作 one-hot 后再映射到 action embedding
        self.action_encoder = nn.Sequential(
            nn.Linear(num_actions, action_embed_dim),
            nn.ELU(),
        )

        # GRU 输入由 previous stochastic state 和 action embedding 拼接得到
        self.rnn_input = nn.Sequential(
            nn.Linear(stoch_dim + action_embed_dim, hidden_dim),
            nn.ELU(),
        )

        # deterministic dynamics
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # prior 网络：只看 deterministic state h_t
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

        # posterior 网络：看 deterministic state h_t 和当前图像 embedding e_t
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embedding_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

        self.apply(init_weights)

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """
        构造初始 RSSM state。

        初始状态一般设为 0：
        - h_0 = 0
        - z_0 = 0
        """
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        mean = torch.zeros(batch_size, self.stoch_dim, device=device)
        std = torch.ones(batch_size, self.stoch_dim, device=device)

        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def action_to_onehot(self, action: torch.Tensor) -> torch.Tensor:
        """
        将动作转换为 one-hot。

        支持两种输入：
        1. action.shape = (B,)        整数动作；
        2. action.shape = (B, A)      已经是 one-hot 或概率形式。
        """
        if action.ndim == 1:
            action = F.one_hot(action.long(), num_classes=self.num_actions).float()
        elif action.ndim == 2:
            if action.shape[-1] != self.num_actions:
                raise ValueError(
                    f"action 最后一维应该是 num_actions={self.num_actions}，"
                    f"但得到 {action.shape}"
                )
            action = action.float()
        else:
            raise ValueError(f"不支持的 action shape: {action.shape}")

        return action

    def _compute_stats(self, raw_stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将网络输出转换为高斯分布参数。

        raw_stats 最后一维大小为 2 * stoch_dim：
        - 前半部分是 mean；
        - 后半部分经过 softplus 得到 std。
        """
        mean, raw_std = torch.chunk(raw_stats, chunks=2, dim=-1)

        # softplus 保证 std 为正，加 min_std 防止标准差过小导致训练不稳定
        std = F.softplus(raw_std) + self.min_std

        return mean, std

    def _sample_stoch(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        使用 reparameterization trick 从高斯分布中采样。
        """
        eps = torch.randn_like(mean)
        return mean + std * eps

    def img_step(self, prev_state: RSSMState, action: torch.Tensor) -> RSSMState:
        """
        Imagination step / prior step。

        只使用：
        - 上一状态 prev_state；
        - 当前动作 action；

        不使用当前观测 embedding。

        该函数用于：
        1. 生成 prior state；
        2. imagination rollout。
        """
        action_onehot = self.action_to_onehot(action)
        action_emb = self.action_encoder(action_onehot)

        x = torch.cat([prev_state.stoch, action_emb], dim=-1)
        x = self.rnn_input(x)

        deter = self.gru(x, prev_state.deter)

        prior_stats = self.prior_net(deter)
        mean, std = self._compute_stats(prior_stats)
        stoch = self._sample_stoch(mean, std)

        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def obs_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        embedding: torch.Tensor,
    ) -> Tuple[RSSMState, RSSMState]:
        """
        Observation step。

        先根据 prev_state 和 action 得到 prior，
        再结合当前 embedding 得到 posterior。

        返回：
            posterior:
                看过当前观测后的状态，训练真实轨迹时使用。

            prior:
                没看当前观测前的预测状态，用于 KL 约束。
        """
        prior = self.img_step(prev_state, action)

        posterior_input = torch.cat([prior.deter, embedding], dim=-1)
        posterior_stats = self.posterior_net(posterior_input)
        mean, std = self._compute_stats(posterior_stats)
        stoch = self._sample_stoch(mean, std)

        posterior = RSSMState(
            deter=prior.deter,
            stoch=stoch,
            mean=mean,
            std=std,
        )

        return posterior, prior

    def observe(self, embeddings: torch.Tensor, actions: torch.Tensor) -> Tuple[RSSMState, RSSMState]:
        """
        对一段真实观测序列执行 observation rollout。

        参数：
            embeddings:
                图像 embedding 序列，形状为 (B, T, embedding_dim)

            actions:
                previous action 序列，形状为 (B, T)
                actions[:, t] 表示进入 obs_t 之前执行的动作。

        返回：
            posteriors:
                posterior state 序列，每个字段形状为 (B, T, dim)

            priors:
                prior state 序列，每个字段形状为 (B, T, dim)
        """
        if embeddings.ndim != 3:
            raise ValueError(
                f"embeddings 应为 (B, T, embedding_dim)，但得到 {embeddings.shape}"
            )

        if actions.ndim != 2:
            raise ValueError(
                f"actions 应为 (B, T)，但得到 {actions.shape}"
            )

        batch_size, seq_len, embed_dim = embeddings.shape

        if embed_dim != self.embedding_dim:
            raise ValueError(
                f"embedding_dim 应为 {self.embedding_dim}，但得到 {embed_dim}"
            )

        if actions.shape[0] != batch_size or actions.shape[1] != seq_len:
            raise ValueError(
                f"actions shape 应与 embeddings 的 B,T 对齐。"
                f"embeddings={embeddings.shape}, actions={actions.shape}"
            )

        state = self.initial_state(batch_size, embeddings.device)

        posterior_states = []
        prior_states = []

        for t in range(seq_len):
            action_t = actions[:, t]
            embedding_t = embeddings[:, t]

            posterior, prior = self.obs_step(
                prev_state=state,
                action=action_t,
                embedding=embedding_t,
            )

            posterior_states.append(posterior)
            prior_states.append(prior)

            # 真实观测序列中，下一步从 posterior 继续
            state = posterior

        posteriors = self.stack_states(posterior_states, dim=1)
        priors = self.stack_states(prior_states, dim=1)

        return posteriors, priors

    def imagine(self, init_state: RSSMState, actions: torch.Tensor) -> RSSMState:
        """
        从某个初始 state 出发，根据动作序列进行 imagination rollout。

        参数：
            init_state:
                imagination 的起点状态，通常来自真实观测序列最后一个 posterior。

            actions:
                未来动作序列，形状为 (B, H)

        返回：
            imagined_states:
                prior state 序列，每个字段形状为 (B, H, dim)
        """
        if actions.ndim != 2:
            raise ValueError(f"actions 应为 (B, H)，但得到 {actions.shape}")

        batch_size, horizon = actions.shape

        if init_state.deter.shape[0] != batch_size:
            raise ValueError(
                f"init_state batch size 与 actions 不匹配："
                f"{init_state.deter.shape[0]} vs {batch_size}"
            )

        state = init_state
        imagined_states = []

        for t in range(horizon):
            action_t = actions[:, t]
            state = self.img_step(state, action_t)
            imagined_states.append(state)

        return self.stack_states(imagined_states, dim=1)

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        """
        将 deterministic state 和 stochastic state 拼接成模型特征。

        后续 RewardModel / Decoder / Actor / Critic 都可以使用这个 feature。

        如果 state 是单步：
            deter: (B, deter_dim)
            stoch: (B, stoch_dim)
            feat:  (B, deter_dim + stoch_dim)

        如果 state 是序列：
            deter: (B, T, deter_dim)
            stoch: (B, T, stoch_dim)
            feat:  (B, T, deter_dim + stoch_dim)
        """
        return torch.cat([state.deter, state.stoch], dim=-1)

    @staticmethod
    def stack_states(states, dim: int) -> RSSMState:
        """
        将多个 RSSMState 沿指定维度堆叠。
        """
        deter = torch.stack([s.deter for s in states], dim=dim)
        stoch = torch.stack([s.stoch for s in states], dim=dim)
        mean = torch.stack([s.mean for s in states], dim=dim)
        std = torch.stack([s.std for s in states], dim=dim)

        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    @staticmethod
    def select_state(state: RSSMState, index: int) -> RSSMState:
        """
        从序列 RSSMState 中选择某个时间步。

        输入字段形状：
            (B, T, dim)

        输出字段形状：
            (B, dim)
        """
        return RSSMState(
            deter=state.deter[:, index],
            stoch=state.stoch[:, index],
            mean=state.mean[:, index],
            std=state.std[:, index],
        )