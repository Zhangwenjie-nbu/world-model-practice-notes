import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class RSSMState:
    h: torch.Tensor
    z: torch.Tensor


class RSSM(nn.Module):

    def __init__(self, deter_dim, stoch_dim, action_dim, hidden_dim, obs_dim, device):
        super().__init__()

        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.obs_dim = obs_dim

        # ===== dynamics =====
        self.input_mlp = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # ===== prior network =====
        # 输入 h_t，输出 (mean, log_std)
        self.prior_mlp = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.prior_mean = nn.Linear(hidden_dim, stoch_dim)
        self.prior_logstd = nn.Linear(hidden_dim, stoch_dim)

        self.post_mlp = nn.Sequential(
            nn.Linear(deter_dim + obs_dim, hidden_dim),  # 注意这里输入包含 obs
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.post_mean = nn.Linear(hidden_dim, stoch_dim)
        self.post_logstd = nn.Linear(hidden_dim, stoch_dim)

    def initial_state(self, batch_size):
        h = torch.zeros(batch_size, self.deter_dim, device=self.device)
        z = torch.zeros(batch_size, self.stoch_dim, device=self.device)
        return RSSMState(h=h, z=z)

    def get_feat(self, state):
        return torch.cat([state.h, state.z], dim=-1)

    def img_step(self, prev_state, prev_action):
        """
        imagination step（现在是完整版本）

        1. 更新 h_t
        2. 用 prior 生成 z_t
        """

        # ===== 1. 更新 h_t =====
        x = torch.cat([prev_state.z, prev_action], dim=-1)
        x = self.input_mlp(x)
        h = self.gru(x, prev_state.h)

        # ===== 2. prior 预测 z_t =====
        z, mean, std = self._prior(h)

        return RSSMState(h=h, z=z), mean, std

    def _prior(self, h):
        """
        根据 h_t 预测 z_t 的分布，并采样。

        输入:
            h: [B, deter_dim]

        输出:
            z:     [B, stoch_dim]
            mean:  [B, stoch_dim]
            std:   [B, stoch_dim]
        """

        x = self.prior_mlp(h)

        mean = self.prior_mean(x)
        log_std = self.prior_logstd(x)

        # 防止数值不稳定（非常重要）
        log_std = torch.clamp(log_std, -5, 2)

        std = torch.exp(log_std)

        # ===== reparameterization =====
        eps = torch.randn_like(mean)
        z = mean + std * eps

        return z, mean, std

    def observe_step(self, prev_state, prev_action, obs):
        """
        单步 posterior 更新（核心函数）

        输入:
            prev_state:
                h_{t-1}, z_{t-1}
            prev_action:
                a_{t-1}
            obs:
                o_t

        输出:
            posterior_state
            prior_mean, prior_std
            post_mean, post_std
        """

        # ===== 1. 先做 dynamics（得到 h_t）=====
        x = torch.cat([prev_state.z, prev_action], dim=-1)
        x = self.input_mlp(x)
        h = self.gru(x, prev_state.h)

        # ===== 2. prior =====
        prior_z, prior_mean, prior_std = self._prior(h)

        # ===== 3. posterior =====
        post_z, post_mean, post_std = self._posterior(h, obs)

        # ===== 4. 返回 posterior state =====
        state = RSSMState(h=h, z=post_z)

        return state, prior_mean, prior_std, post_mean, post_std


    def _posterior(self, h, obs):
        """
        posterior: q(z_t | h_t, o_t)

        输入:
            h:   [B, deter_dim]
            obs: [B, obs_dim]

        输出:
            z, mean, std
        """

        # ===== 1. 拼接输入 =====
        x = torch.cat([h, obs], dim=-1)

        # ===== 2. MLP =====
        x = self.post_mlp(x)

        # ===== 3. 输出分布参数 =====
        mean = self.post_mean(x)
        log_std = self.post_logstd(x)

        log_std = torch.clamp(log_std, -5, 2)
        std = torch.exp(log_std)

        # ===== 4. reparameterization =====
        eps = torch.randn_like(mean)
        z = mean + std * eps

        return z, mean, std


    def observe_rollout(self, obs, actions):
        """
        对一整段序列执行 RSSM 前向。

        输入:
            obs:     [B, T, obs_dim]
            actions: [B, T, action_dim]

        输出:
            states:        list of RSSMState（长度 T）
            prior_means:   [B, T, stoch_dim]
            prior_stds:    [B, T, stoch_dim]
            post_means:    [B, T, stoch_dim]
            post_stds:     [B, T, stoch_dim]
        """

        B, T, _ = obs.shape

        # ===== 初始化 =====
        state = self.initial_state(B)

        # 存储容器
        states = []
        prior_means = []
        prior_stds = []
        post_means = []
        post_stds = []

        # t=0 时的 prev_action = 0
        prev_action = torch.zeros(B, self.action_dim, device=obs.device)

        for t in range(T):
            obs_t = obs[:, t]
            action_t = actions[:, t]

            # ===== 核心更新 =====
            state, p_mean, p_std, q_mean, q_std = self.observe_step(
                state, prev_action, obs_t
            )

            # ===== 存储 =====
            states.append(state)
            prior_means.append(p_mean)
            prior_stds.append(p_std)
            post_means.append(q_mean)
            post_stds.append(q_std)

            # 更新 prev_action
            prev_action = action_t

        # ===== stack =====
        prior_means = torch.stack(prior_means, dim=1)
        prior_stds = torch.stack(prior_stds, dim=1)
        post_means = torch.stack(post_means, dim=1)
        post_stds = torch.stack(post_stds, dim=1)

        return states, prior_means, prior_stds, post_means, post_stds

    def get_feat_seq(self, states):
        """
        把 state 序列转换为 feature tensor

        输入:
            states: list of RSSMState（长度 T）

        输出:
            feat: [B, T, deter_dim + stoch_dim]
        """

        feats = [self.get_feat(s) for s in states]
        return torch.stack(feats, dim=1)