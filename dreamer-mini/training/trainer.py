import torch

from training.imagination import imagine_rollout
from training.losses import (
    world_model_loss,
    compute_lambda_return,
    value_loss,
    actor_loss_with_entropy,
)


class DreamerTrainer:
    """
    Dreamer-mini 的训练器。

    当前版本职责：
        1. 更新 world model
        2. 做 imagination rollout
        3. 更新 value
        4. 更新 actor
    """

    def __init__(
        self,
        rssm,
        obs_head,
        reward_head,
        actor,
        value_net,
        replay_buffer,
        wm_optimizer,
        actor_optimizer,
        value_optimizer,
        kl_loss_fn,
        imagine_horizon,
        gamma=0.99,
        lambda_=0.95,
        entropy_coeff=0.01,
    ):
        self.rssm = rssm
        self.obs_head = obs_head
        self.reward_head = reward_head
        self.actor = actor
        self.value_net = value_net
        self.replay_buffer = replay_buffer

        self.wm_optimizer = wm_optimizer
        self.actor_optimizer = actor_optimizer
        self.value_optimizer = value_optimizer

        self.kl_loss_fn = kl_loss_fn
        self.imagine_horizon = imagine_horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coeff = entropy_coeff

    def train_step(self, batch_size):
        """
        执行一次完整 Dreamer 训练。

        返回:
            metrics: 训练指标字典
        """
        # =====================================================
        # 1. 从 replay buffer 采样真实序列
        # =====================================================
        batch = self.replay_buffer.sample_batch(batch_size)

        obs = batch["obs"]           # [B, T, obs_dim]
        actions = batch["actions"]   # [B, T, action_dim]
        rewards = batch["rewards"]   # [B, T]

        # =====================================================
        # 2. 更新 world model
        # =====================================================
        self.rssm.train()
        self.obs_head.train()
        self.reward_head.train()

        wm_loss, wm_metrics, wm_outputs = world_model_loss(
            rssm=self.rssm,
            obs_head=self.obs_head,
            reward_head=self.reward_head,
            obs=obs,
            actions=actions,
            rewards=rewards,
            kl_loss_fn=self.kl_loss_fn,
        )

        self.wm_optimizer.zero_grad()
        wm_loss.backward()
        self.wm_optimizer.step()

        # =====================================================
        # 3. 取 posterior rollout 的最后一个 state 作为 imagination 起点
        # =====================================================
        start_states = wm_outputs["states"]

        # 这里的 imagination 通常不希望 actor/value 反向穿回 world model，
        # 所以先 detach 起点 state
        start_state = start_states[-1]
        start_state = type(start_state)(
            h=start_state.h.detach(),
            z=start_state.z.detach(),
        )

        detached_start_states = [start_state]

        # =====================================================
        # 4. imagination rollout
        # =====================================================
        imag_feats, imag_actions, imag_rewards = imagine_rollout(
            rssm=self.rssm,
            actor=self.actor,
            reward_head=self.reward_head,
            start_states=detached_start_states,
            horizon=self.imagine_horizon,
        )
        # imag_feats:   [B, H, feat_dim]
        # imag_rewards: [B, H, 1]

        # =====================================================
        # 5. 更新 value
        # =====================================================
        self.value_net.train()

        imag_values = self.value_net(imag_feats)  # [B, H, 1]

        lambda_returns = compute_lambda_return(
            rewards=imag_rewards,
            values=imag_values,
            gamma=self.gamma,
            lambda_=self.lambda_,
        ).detach()  # target 不反传

        v_loss = value_loss(
            value_net=self.value_net,
            feat=imag_feats.detach(),
            lambda_returns=lambda_returns,
        )

        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()

        # =====================================================
        # 6. 更新 actor
        # =====================================================
        # 重新算一次 imagined trajectory，避免图被上一步消耗/污染
        imag_feats_actor, _, imag_rewards_actor = imagine_rollout(
            rssm=self.rssm,
            actor=self.actor,
            reward_head=self.reward_head,
            start_states=detached_start_states,
            horizon=self.imagine_horizon,
        )

        imag_values_actor = self.value_net(imag_feats_actor)

        actor_returns = compute_lambda_return(
            rewards=imag_rewards_actor,
            values=imag_values_actor,
            gamma=self.gamma,
            lambda_=self.lambda_,
        )

        a_loss = actor_loss_with_entropy(
            lambda_returns=actor_returns,
            entropy=None,  # 如果你使用了 entropy
            entropy_coeff=0.01
)

        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        # =====================================================
        # 7. 汇总指标
        # =====================================================
        metrics = {
            **wm_metrics,
            "value_loss": v_loss.item(),
            "actor_loss": a_loss.item(),
            "imag_reward_mean": imag_rewards.mean().item(),
            "imag_value_mean": imag_values.mean().item(),
            "lambda_return_mean": lambda_returns.mean().item(),
        }

        return metrics