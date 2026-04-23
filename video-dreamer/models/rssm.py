from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSSMState:
    """
    RSSM 单个时间步的状态。

    字段说明：
        deter:
            确定性隐状态 h_t，shape 为 [B, deter_dim]

        stoch:
            随机隐变量 z_t，shape 为 [B, stoch_dim]

        prior_mean:
            prior 分布均值，shape 为 [B, stoch_dim]

        prior_std:
            prior 分布标准差，shape 为 [B, stoch_dim]

        post_mean:
            posterior 分布均值，shape 为 [B, stoch_dim]
            在 imagine 阶段为 None。

        post_std:
            posterior 分布标准差，shape 为 [B, stoch_dim]
            在 imagine 阶段为 None。
    """

    deter: torch.Tensor
    stoch: torch.Tensor
    prior_mean: torch.Tensor
    prior_std: torch.Tensor
    post_mean: Optional[torch.Tensor] = None
    post_std: Optional[torch.Tensor] = None


@dataclass
class RSSMRollout:
    """
    RSSM 对一整段序列 rollout 后的结果。

    字段说明：
        states:
            每个时间步的 RSSMState 列表，长度为 T。

        features:
            每个时间步的 decoder 输入特征 [h_t, z_t]，
            shape 为 [B, T, deter_dim + stoch_dim]。

        prior_means:
            prior 均值序列，shape 为 [B, T, stoch_dim]。

        prior_stds:
            prior 标准差序列，shape 为 [B, T, stoch_dim]。

        post_means:
            posterior 均值序列，shape 为 [B, T, stoch_dim]。
            对 imagine_rollout 来说，该字段为 None。

        post_stds:
            posterior 标准差序列，shape 为 [B, T, stoch_dim]。
            对 imagine_rollout 来说，该字段为 None。

        final_state:
            rollout 结束后的最后一个状态。
            后续做未来预测时，通常会从这个状态继续 imagine。
    """

    states: List[RSSMState]
    features: torch.Tensor
    prior_means: torch.Tensor
    prior_stds: torch.Tensor
    post_means: Optional[torch.Tensor]
    post_stds: Optional[torch.Tensor]
    final_state: RSSMState


class RSSM(nn.Module):
    """
    视频世界模型中的 RSSM 模块。

    作用：
        根据历史状态递推 deterministic state，
        并构造 stochastic latent 的 prior/posterior 分布。

    主要接口：
        init_state:
            初始化 h_0 和 z_0。

        observe_step:
            给定上一状态和当前观测 embedding，执行一个 posterior 更新步。

        imagine_step:
            给定上一状态，不使用观测 embedding，只用 prior 执行一个想象步。

        observe_rollout:
            对整段观测 embedding 序列执行 posterior rollout。

        imagine_rollout:
            从某个状态出发，使用 prior 向未来 rollout 若干步。

        get_feature:
            将 h_t 和 z_t 拼接成 decoder 输入 feature。
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        deter_dim: int = 128,
        stoch_dim: int = 32,
        hidden_dim: int = 256,
        min_std: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std

        # GRUCell 用于递推确定性隐状态 h_t。
        #
        # 输入是上一时刻随机隐变量 z_{t-1}。
        # 隐状态是上一时刻确定性状态 h_{t-1}。
        self.gru = nn.GRUCell(
            input_size=stoch_dim,
            hidden_size=deter_dim,
        )

        # prior 网络：
        # 输入 h_t
        # 输出 prior_mean 和 prior_std 的原始参数
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

        # posterior 网络：
        # 输入 [h_t, e_t]
        # 输出 post_mean 和 post_std 的原始参数
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

    def init_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """
        初始化 RSSM 状态。

        初始状态设为全零：
            h_0 = 0
            z_0 = 0
        """
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=device)

        prior_mean = torch.zeros(batch_size, self.stoch_dim, device=device)
        prior_std = torch.ones(batch_size, self.stoch_dim, device=device)

        post_mean = torch.zeros(batch_size, self.stoch_dim, device=device)
        post_std = torch.ones(batch_size, self.stoch_dim, device=device)

        return RSSMState(
            deter=deter,
            stoch=stoch,
            prior_mean=prior_mean,
            prior_std=prior_std,
            post_mean=post_mean,
            post_std=post_std,
        )

    def _compute_stats(self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将网络输出参数转换为高斯分布的 mean 和 std。

        params:
            shape 为 [B, 2 * stoch_dim]。

        返回：
            mean: [B, stoch_dim]
            std:  [B, stoch_dim]
        """
        mean, raw_std = torch.chunk(params, chunks=2, dim=-1)

        # softplus 保证标准差为正，min_std 防止标准差过小
        std = F.softplus(raw_std) + self.min_std

        return mean, std

    def _sample_normal(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        使用重参数化技巧从高斯分布中采样。

        z = mean + std * eps
        eps ~ N(0, I)
        """
        eps = torch.randn_like(mean)
        sample = mean + std * eps
        return sample

    def _prior(self, deter: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据确定性状态 h_t 计算 prior 分布。

        p(z_t | h_t)
        """
        prior_params = self.prior_net(deter)
        prior_mean, prior_std = self._compute_stats(prior_params)
        return prior_mean, prior_std

    def _posterior(
        self,
        deter: torch.Tensor,
        embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据确定性状态 h_t 和当前观测 embedding e_t 计算 posterior 分布。

        q(z_t | h_t, e_t)
        """
        x = torch.cat([deter, embed], dim=-1)
        post_params = self.posterior_net(x)
        post_mean, post_std = self._compute_stats(post_params)
        return post_mean, post_std

    def observe_step(
        self,
        prev_state: RSSMState,
        embed: torch.Tensor,
    ) -> RSSMState:
        """
        执行一个观测更新步。

        训练阶段使用该函数。
        它会使用当前真实观测 embedding e_t 来计算 posterior。

        计算：
            h_t = GRU(z_{t-1}, h_{t-1})
            prior = p(z_t | h_t)
            posterior = q(z_t | h_t, e_t)
            z_t ~ posterior
        """
        if embed.ndim != 2:
            raise ValueError(
                f"observe_step 期望 embed shape 为 [B, embedding_dim]，但得到 {embed.shape}"
            )

        # 递推确定性状态 h_t
        deter = self.gru(prev_state.stoch, prev_state.deter)

        # 根据 h_t 计算 prior
        prior_mean, prior_std = self._prior(deter)

        # 根据 h_t 和当前观测 embedding 计算 posterior
        post_mean, post_std = self._posterior(deter, embed)

        # 训练观测阶段从 posterior 采样 z_t
        stoch = self._sample_normal(post_mean, post_std)

        return RSSMState(
            deter=deter,
            stoch=stoch,
            prior_mean=prior_mean,
            prior_std=prior_std,
            post_mean=post_mean,
            post_std=post_std,
        )

    def imagine_step(self, prev_state: RSSMState) -> RSSMState:
        """
        执行一个想象步。

        预测未来阶段使用该函数。
        它不使用真实观测 embedding，只依赖 prior。

        计算：
            h_t = GRU(z_{t-1}, h_{t-1})
            prior = p(z_t | h_t)
            z_t ~ prior
        """
        # 递推确定性状态 h_t
        deter = self.gru(prev_state.stoch, prev_state.deter)

        # 根据 h_t 计算 prior
        prior_mean, prior_std = self._prior(deter)

        # 未来想象阶段从 prior 采样 z_t
        stoch = self._sample_normal(prior_mean, prior_std)

        return RSSMState(
            deter=deter,
            stoch=stoch,
            prior_mean=prior_mean,
            prior_std=prior_std,
            post_mean=None,
            post_std=None,
        )

    def get_feature(self, state: RSSMState) -> torch.Tensor:
        """
        将 RSSM 状态转换成 Decoder 输入 feature。

        feature = concat(h_t, z_t)
        """
        feature = torch.cat([state.deter, state.stoch], dim=-1)
        return feature

    def observe_rollout(
        self,
        embeds: torch.Tensor,
        init_state: Optional[RSSMState] = None,
    ) -> RSSMRollout:
        """
        对整段观测 embedding 序列执行 posterior rollout。

        参数：
            embeds:
                Encoder 输出的 embedding 序列，
                shape 为 [B, T, embedding_dim]。

            init_state:
                初始 RSSMState。
                如果为 None，则自动初始化为全零状态。

        返回：
            RSSMRollout，包含整段状态、features、prior 参数和 posterior 参数。

        计算流程：
            对 t = 1 ... T：
                state_t = observe_step(state_{t-1}, embed_t)
        """
        if embeds.ndim != 3:
            raise ValueError(
                f"observe_rollout 期望 embeds shape 为 [B, T, embedding_dim]，但得到 {embeds.shape}"
            )

        B, T, D = embeds.shape

        if D != self.embedding_dim:
            raise ValueError(
                f"observe_rollout 输入 embedding_dim 不匹配，"
                f"模型期望 {self.embedding_dim}，实际得到 {D}"
            )

        device = embeds.device

        if init_state is None:
            state = self.init_state(batch_size=B, device=device)
        else:
            state = init_state

        states = []
        features = []
        prior_means = []
        prior_stds = []
        post_means = []
        post_stds = []

        for t in range(T):
            embed_t = embeds[:, t]  # [B, embedding_dim]

            state = self.observe_step(
                prev_state=state,
                embed=embed_t,
            )

            states.append(state)
            features.append(self.get_feature(state))
            prior_means.append(state.prior_mean)
            prior_stds.append(state.prior_std)
            post_means.append(state.post_mean)
            post_stds.append(state.post_std)

        # 将 list 形式的时间序列堆叠成标准张量
        features = torch.stack(features, dim=1)          # [B, T, deter_dim + stoch_dim]
        prior_means = torch.stack(prior_means, dim=1)    # [B, T, stoch_dim]
        prior_stds = torch.stack(prior_stds, dim=1)      # [B, T, stoch_dim]
        post_means = torch.stack(post_means, dim=1)      # [B, T, stoch_dim]
        post_stds = torch.stack(post_stds, dim=1)        # [B, T, stoch_dim]

        return RSSMRollout(
            states=states,
            features=features,
            prior_means=prior_means,
            prior_stds=prior_stds,
            post_means=post_means,
            post_stds=post_stds,
            final_state=state,
        )

    def imagine_rollout(
        self,
        start_state: RSSMState,
        horizon: int,
    ) -> RSSMRollout:
        """
        从给定状态出发，使用 prior 向未来 rollout 若干步。

        参数：
            start_state:
                起始状态。
                通常来自 context observe_rollout 的 final_state。

            horizon:
                想象未来的步数。

        返回：
            RSSMRollout。
            其中 post_means 和 post_stds 为 None。

        计算流程：
            对 t = 1 ... horizon：
                state_t = imagine_step(state_{t-1})
        """
        if horizon <= 0:
            raise ValueError(f"horizon 必须大于 0，但得到 {horizon}")

        state = start_state

        states = []
        features = []
        prior_means = []
        prior_stds = []

        for _ in range(horizon):
            state = self.imagine_step(prev_state=state)

            states.append(state)
            features.append(self.get_feature(state))
            prior_means.append(state.prior_mean)
            prior_stds.append(state.prior_std)

        features = torch.stack(features, dim=1)          # [B, H, deter_dim + stoch_dim]
        prior_means = torch.stack(prior_means, dim=1)    # [B, H, stoch_dim]
        prior_stds = torch.stack(prior_stds, dim=1)      # [B, H, stoch_dim]

        return RSSMRollout(
            states=states,
            features=features,
            prior_means=prior_means,
            prior_stds=prior_stds,
            post_means=None,
            post_stds=None,
            final_state=state,
        )