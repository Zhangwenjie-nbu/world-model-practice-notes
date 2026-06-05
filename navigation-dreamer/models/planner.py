# models/planner.py

from dataclasses import dataclass
from typing import Optional

import torch

from models.rssm import RSSM, RSSMState
from models.actor import DiscreteActor
from models.reward_model import RewardModel
from models.critic import Critic


# Current discrete action mapping.
ACTION_FORWARD = 0
ACTION_BACKWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3


@dataclass
class ActorImaginationOutput:
    """
    Actor-driven imagination rollout 的输出。
    """
    states: RSSMState
    actions: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    entropies: torch.Tensor
    returns: torch.Tensor
    terminal_value: torch.Tensor


@dataclass
class ActionSequenceEvaluation:
    """
    随机动作序列评估结果。
    """
    candidate_actions: torch.Tensor
    imagined_states: RSSMState
    rewards: torch.Tensor
    terminal_values: torch.Tensor
    returns: torch.Tensor
    sequence_scores: torch.Tensor
    best_index: torch.Tensor
    best_actions: torch.Tensor
    best_action: torch.Tensor


@dataclass
class ContinueAwareActionSequenceEvaluation:
    """
    Continue-aware random shooting 评估结果。
    """
    candidate_actions: torch.Tensor
    imagined_states: RSSMState
    rewards: torch.Tensor
    continues: torch.Tensor
    terminal_values: torch.Tensor
    returns: torch.Tensor
    sequence_scores: torch.Tensor
    best_index: torch.Tensor
    best_actions: torch.Tensor
    best_action: torch.Tensor


def compute_bootstrapped_returns(
    rewards: torch.Tensor,
    terminal_value: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    计算 bootstrapped return。

    参数：
        rewards:
            imagined rewards，形状为 (B, H, 1)。

        terminal_value:
            最后一个 imagined state 的 value，形状为 (B, 1)。

        gamma:
            折扣因子。

    返回：
        returns:
            每个 imagined step 的 return，形状为 (B, H, 1)。

    计算方式：
        G_{H-1} = r_H + gamma * V(s_H)
        G_{t}   = r_{t+1} + gamma * G_{t+1}
    """

    if rewards.ndim != 3:
        raise ValueError(f"rewards 应为 (B, H, 1)，但得到 {rewards.shape}")

    if terminal_value.ndim != 2:
        raise ValueError(
            f"terminal_value 应为 (B, 1)，但得到 {terminal_value.shape}"
        )

    batch_size, horizon, reward_dim = rewards.shape

    if reward_dim != 1:
        raise ValueError(f"rewards 最后一维应该为 1，但得到 {reward_dim}")

    if terminal_value.shape != (batch_size, 1):
        raise ValueError(
            f"terminal_value shape 应为 {(batch_size, 1)}，"
            f"但得到 {terminal_value.shape}"
        )

    returns = torch.zeros_like(rewards)

    running_return = terminal_value

    for t in reversed(range(horizon)):
        running_return = rewards[:, t] + gamma * running_return
        returns[:, t] = running_return

    return returns


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continues: torch.Tensor,
    bootstrap: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """
    计算 continue-aware lambda return。

    rewards:
        (B, H, 1)

    values:
        (B, H, 1)

    continues:
        (B, H, 1)，取值范围 [0, 1]

    bootstrap:
        (B, 1)

    返回：
        returns:
            (B, H, 1)
    """

    if rewards.shape != values.shape:
        raise ValueError(f"rewards 和 values shape 不一致: {rewards.shape} vs {values.shape}")

    if rewards.shape != continues.shape:
        raise ValueError(f"rewards 和 continues shape 不一致: {rewards.shape} vs {continues.shape}")

    batch_size, horizon, reward_dim = rewards.shape

    if reward_dim != 1:
        raise ValueError(f"reward_dim 应为 1，但得到 {reward_dim}")

    if bootstrap.shape != (batch_size, 1):
        raise ValueError(f"bootstrap shape 应为 {(batch_size, 1)}，但得到 {bootstrap.shape}")

    returns = torch.zeros_like(rewards)

    next_return = bootstrap

    for t in reversed(range(horizon)):
        if t == horizon - 1:
            next_value = bootstrap
        else:
            next_value = values[:, t + 1]

        target = rewards[:, t] + gamma * continues[:, t] * (
            (1.0 - lambda_) * next_value
            + lambda_ * next_return
        )

        returns[:, t] = target
        next_return = target

    return returns


def repeat_state(state: RSSMState, repeats: int) -> RSSMState:
    """
    将单个 batch 的 RSSMState 复制 repeats 次。

    常用于 random shooting：
        当前只有一个 state，但需要并行评估多条候选动作序列。

    输入：
        state 中每个字段形状为 (B, dim)

    输出：
        每个字段形状为 (B * repeats, dim)
    """

    if repeats <= 0:
        raise ValueError(f"repeats 必须大于 0，但得到 {repeats}")

    return RSSMState(
        deter=state.deter.repeat_interleave(repeats, dim=0),
        stoch=state.stoch.repeat_interleave(repeats, dim=0),
        mean=state.mean.repeat_interleave(repeats, dim=0),
        std=state.std.repeat_interleave(repeats, dim=0),
    )


def actor_imagination_rollout(
    rssm: RSSM,
    actor: DiscreteActor,
    reward_model: RewardModel,
    continue_model,
    critic: Critic,
    init_state: RSSMState,
    horizon: int = 8,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    deterministic: bool = False,
    temperature: float = 1.0,
) -> ActorImaginationOutput:
    """
    使用 Actor 在 latent space 中进行 imagination rollout。

    当前版本新增：
        ContinueModel，用于预测 imagined state 是否继续。
    """

    if horizon <= 0:
        raise ValueError(f"horizon 必须大于 0，但得到 {horizon}")

    state = init_state

    imagined_states = []
    actions = []
    rewards = []
    continues = []
    values = []
    log_probs = []
    entropies = []

    for _ in range(horizon):
        actor_feat = torch.cat([state.deter, state.mean], dim=-1)

        action, log_prob, entropy = actor.sample_action(
            feat=actor_feat,
            deterministic=deterministic,
            temperature=temperature,
        )

        next_state = rssm.img_step(
            prev_state=state,
            action=action,
        )

        next_feat = rssm.get_feat(next_state)

        reward = reward_model(next_feat)
        continue_prob = continue_model.continue_prob(next_feat)
        value = critic(next_feat)

        imagined_states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        continues.append(continue_prob)
        values.append(value)
        log_probs.append(log_prob)
        entropies.append(entropy)

        state = next_state

    imagined_states = RSSM.stack_states(imagined_states, dim=1)

    actions = torch.stack(actions, dim=1)
    rewards = torch.stack(rewards, dim=1)
    continues = torch.stack(continues, dim=1)
    values = torch.stack(values, dim=1)
    log_probs = torch.stack(log_probs, dim=1)
    entropies = torch.stack(entropies, dim=1)

    terminal_feat = rssm.get_feat(state)
    terminal_value = critic(terminal_feat)

    returns = compute_lambda_returns(
        rewards=rewards,
        values=values,
        continues=continues,
        bootstrap=terminal_value,
        gamma=gamma,
        lambda_=lambda_,
    )

    return ActorImaginationOutput(
        states=imagined_states,
        actions=actions,
        rewards=rewards,
        continues=continues,
        values=values,
        log_probs=log_probs,
        entropies=entropies,
        returns=returns,
        terminal_value=terminal_value,
    )


def select_action_by_random_shooting(
    rssm: RSSM,
    reward_model: RewardModel,
    critic: Critic,
    init_state: RSSMState,
    num_actions: int,
    horizon: int = 8,
    num_candidates: int = 64,
    gamma: float = 0.99,
    generator: Optional[torch.Generator] = None,
) -> ActionSequenceEvaluation:
    """
    使用 random shooting 进行动作选择。

    方法：
        1. 随机生成 num_candidates 条动作序列；
        2. 对每条动作序列执行 RSSM imagination rollout；
        3. 用 RewardModel + Critic 计算 return；
        4. 选择 return 最大的动作序列；
        5. 返回该序列的第一个动作。

    当前实现假设 init_state 的 batch size 为 1。
    """

    if horizon <= 0:
        raise ValueError(f"horizon 必须大于 0，但得到 {horizon}")

    if num_candidates <= 0:
        raise ValueError(f"num_candidates 必须大于 0，但得到 {num_candidates}")

    batch_size = init_state.deter.shape[0]

    if batch_size != 1:
        raise ValueError(
            "当前 random shooting 版本只支持 batch_size=1。"
            f"但得到 batch_size={batch_size}"
        )

    device = init_state.deter.device

    candidate_actions = torch.randint(
        low=0,
        high=num_actions,
        size=(num_candidates, horizon),
        device=device,
        generator=generator,
    )

    expanded_state = repeat_state(init_state, repeats=num_candidates)

    imagined_states = rssm.imagine(
        init_state=expanded_state,
        actions=candidate_actions,
    )

    imagined_feat = rssm.get_feat(imagined_states)

    rewards = reward_model(imagined_feat)

    terminal_state = RSSM.select_state(imagined_states, index=-1)
    terminal_feat = rssm.get_feat(terminal_state)
    terminal_values = critic(terminal_feat)

    returns = compute_bootstrapped_returns(
        rewards=rewards,
        terminal_value=terminal_values,
        gamma=gamma,
    )

    # 每条候选动作序列的分数取第 0 步 return
    sequence_scores = returns[:, 0, 0]

    best_index = torch.argmax(sequence_scores)
    best_actions = candidate_actions[best_index]
    best_action = best_actions[0]

    return ActionSequenceEvaluation(
        candidate_actions=candidate_actions,
        imagined_states=imagined_states,
        rewards=rewards,
        terminal_values=terminal_values,
        returns=returns,
        sequence_scores=sequence_scores,
        best_index=best_index,
        best_actions=best_actions,
        best_action=best_action,
    )


def select_action_by_random_shooting_continue(
    rssm: RSSM,
    reward_model: RewardModel,
    continue_model,
    critic: Critic,
    init_state: RSSMState,
    num_actions: int,
    horizon: int = 12,
    num_candidates: int = 512,
    gamma: float = 0.99,
    action_repeat_penalty: float = 0.03,
    done_risk_penalty: float = 3.0,
    terminal_value_scale: float = 0.35,
    event_model=None,
    success_bonus: float = 8.0,
    collision_penalty: float = 6.0,
    stuck_penalty: float = 5.0,
    generator: Optional[torch.Generator] = None,
) -> ContinueAwareActionSequenceEvaluation:
    """
    Continue-aware random shooting planner。

    方法：
        1. 随机生成 num_candidates 条动作序列；
        2. 使用 RSSM prior rollout；
        3. 用 RewardModel 预测 reward；
        4. 用 ContinueModel 预测 continue；
        5. 用 Critic 估计 terminal value；
        6. 计算 continue-aware return；
        7. 选择 return 最大的动作序列；
        8. 返回该序列第一个动作。

    额外加入 action_repeat_penalty：
        避免序列中 turn_left / turn_right 过多。
        这不是环境 reward，而是 planner 内部的小正则，
        用于防止 planner 选择原地转圈序列。
    """

    if horizon <= 0:
        raise ValueError(f"horizon 必须大于 0，但得到 {horizon}")

    if num_candidates <= 0:
        raise ValueError(f"num_candidates 必须大于 0，但得到 {num_candidates}")

    batch_size = init_state.deter.shape[0]

    if batch_size != 1:
        raise ValueError(
            "当前 random shooting planner 只支持 batch_size=1，"
            f"但得到 batch_size={batch_size}"
        )

    device = init_state.deter.device

    candidate_actions = torch.randint(
        low=0,
        high=num_actions,
        size=(num_candidates, horizon),
        device=device,
        generator=generator,
    )

    # 手动加入一些结构化候选序列，提升搜索质量
    structured = []

    # 连续 forward
    structured.append(torch.full((horizon,), ACTION_FORWARD, device=device, dtype=torch.long))

    # left + forward
    if horizon >= 2:
        seq = torch.full((horizon,), ACTION_FORWARD, device=device, dtype=torch.long)
        seq[0] = ACTION_TURN_LEFT
        structured.append(seq)

    # right + forward
    if horizon >= 2:
        seq = torch.full((horizon,), ACTION_FORWARD, device=device, dtype=torch.long)
        seq[0] = ACTION_TURN_RIGHT
        structured.append(seq)

    # left, left, forward
    if horizon >= 3:
        seq = torch.full((horizon,), ACTION_FORWARD, device=device, dtype=torch.long)
        seq[0] = ACTION_TURN_LEFT
        seq[1] = ACTION_TURN_LEFT
        structured.append(seq)

    # right, right, forward
    if horizon >= 3:
        seq = torch.full((horizon,), ACTION_FORWARD, device=device, dtype=torch.long)
        seq[0] = ACTION_TURN_RIGHT
        seq[1] = ACTION_TURN_RIGHT
        structured.append(seq)

    if len(structured) > 0:
        structured_actions = torch.stack(structured, dim=0)
        n_structured = min(structured_actions.shape[0], num_candidates)
        candidate_actions[:n_structured] = structured_actions[:n_structured]

    expanded_state = repeat_state(
        init_state,
        repeats=num_candidates,
    )

    imagined_states = rssm.imagine(
        init_state=expanded_state,
        actions=candidate_actions,
    )

    imagined_feat = rssm.get_feat(imagined_states)

    rewards = reward_model(imagined_feat)
    continues = continue_model.continue_prob(imagined_feat)

    if event_model is not None:
        event_probs = torch.sigmoid(event_model(imagined_feat))

        success_prob = event_probs[..., 0:1]
        collision_prob = event_probs[..., 1:2]
        stuck_prob = event_probs[..., 2:3]

        # 用显式事件做温和 shaping，避免 event head 主导全部打分。
        rewards = (
            rewards
            + success_bonus * success_prob
            - collision_penalty * collision_prob
            - stuck_penalty * stuck_prob
        )

        # 额外抑制“高碰撞风险但短期 reward 看起来还行”的序列。
        continues = continues * (1.0 - 0.5 * collision_prob) * (1.0 - 0.3 * stuck_prob)

    terminal_state = RSSM.select_state(
        imagined_states,
        index=-1,
    )

    terminal_feat = rssm.get_feat(terminal_state)
    raw_terminal_values = critic(terminal_feat)

    # 终端价值只给中等权重，避免 critic 噪声主导，但也不再完全忽略长程信息。
    terminal_values = terminal_value_scale * raw_terminal_values

    returns = torch.zeros_like(rewards)

    running_return = terminal_values

    for t in reversed(range(horizon)):
        # continue 越低，说明模型认为越可能终止。
        # 由于当前没有区分 success/collision/stuck，先给终止风险一个小惩罚，
        # 避免 planner 主动选择容易终止的动作序列。
        done_risk = 1.0 - continues[:, t]

        running_return = (
            rewards[:, t]
            - done_risk_penalty * done_risk
            + gamma * continues[:, t] * running_return
        )

        returns[:, t] = running_return

    sequence_scores = returns[:, 0, 0]

    # turn regularization
    # 惩罚过多不改变位置的转向动作：left/right
    if action_repeat_penalty > 0.0:
        turn_action = (candidate_actions >= ACTION_TURN_LEFT).float()
        repeat_penalty = turn_action.sum(dim=1) * action_repeat_penalty
        sequence_scores = sequence_scores - repeat_penalty

    best_index = torch.argmax(sequence_scores)
    best_actions = candidate_actions[best_index]
    best_action = best_actions[0]

    return ContinueAwareActionSequenceEvaluation(
        candidate_actions=candidate_actions,
        imagined_states=imagined_states,
        rewards=rewards,
        continues=continues,
        terminal_values=terminal_values,
        returns=returns,
        sequence_scores=sequence_scores,
        best_index=best_index,
        best_actions=best_actions,
        best_action=best_action,
    )