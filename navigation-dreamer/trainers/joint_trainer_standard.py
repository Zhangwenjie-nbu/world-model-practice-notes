# trainers/joint_trainer_standard.py

"""
Standard world-model RL trainer utilities.

This file intentionally removes expert BC / DAgger training from the training path.
It keeps only:
1. World model training:
   Encoder + RSSM + Decoder + RewardModel + ContinueModel
2. Imagination-based Actor-Critic training:
   Actor + Critic updated from latent imagined rollouts

No expert action labels are used here.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.actor import DiscreteActor
from models.continue_model import ContinueModel
from models.critic import Critic
from models.decoder import VisualDecoder
from models.encoder import VisualEncoder
from models.event_model import EventModel
from models.planner import actor_imagination_rollout, compute_lambda_returns
from models.reward_model import RewardModel
from models.rssm import RSSM, RSSMState


@dataclass
class WorldModelLossOutput:
    total_loss: torch.Tensor
    recon_loss: torch.Tensor
    reward_loss: torch.Tensor
    continue_loss: torch.Tensor
    event_loss: torch.Tensor
    kl_loss: torch.Tensor
    continue_acc: torch.Tensor
    event_acc: torch.Tensor
    reconstructed: torch.Tensor
    pred_rewards: torch.Tensor
    pred_continue_logits: torch.Tensor
    pred_event_logits: torch.Tensor
    posteriors: RSSMState
    priors: RSSMState
    posterior_feat: torch.Tensor
    prior_feat: torch.Tensor
    grad_norm: float = 0.0


@dataclass
class ActorCriticLossOutput:
    total_loss: torch.Tensor
    actor_loss: torch.Tensor
    critic_loss: torch.Tensor
    imagined_critic_loss: torch.Tensor
    real_critic_loss: torch.Tensor
    entropy_loss: torch.Tensor
    mean_return: torch.Tensor
    mean_reward: torch.Tensor
    mean_value: torch.Tensor
    mean_continue: torch.Tensor
    action_histogram: torch.Tensor
    grad_norm_actor: float = 0.0
    grad_norm_critic: float = 0.0


@dataclass
class ActorBCLossOutput:
    bc_loss: torch.Tensor
    bc_acc: torch.Tensor
    action_histogram: torch.Tensor
    grad_norm_actor: float = 0.0


def encode_sequence(
    encoder: VisualEncoder,
    obs_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Encode a batch of image sequences.

    Args:
        obs_tensor: (B, T, C, H, W), values in [0, 1]

    Returns:
        embeddings: (B, T, embedding_dim)
    """
    if obs_tensor.ndim != 5:
        raise ValueError(
            f"obs_tensor should be (B, T, C, H, W), got {obs_tensor.shape}"
        )

    batch_size, seq_len, channels, height, width = obs_tensor.shape

    flat_obs = obs_tensor.reshape(batch_size * seq_len, channels, height, width)
    flat_embeddings = encoder(flat_obs)

    return flat_embeddings.reshape(batch_size, seq_len, -1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute masked mean.

    Args:
        values: (B, T, 1)
        mask:   (B, T, 1), 1 for valid steps, 0 for padding
    """
    if values.shape != mask.shape:
        raise ValueError(
            f"values and mask should have same shape, got {values.shape} vs {mask.shape}"
        )

    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def masked_weighted_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute masked weighted mean.

    Args:
        values:  (B, T, 1)
        mask:    (B, T, 1), 1 for valid steps, 0 for padding
        weights: (B, T, 1), non-negative per-step weights
    """
    if values.shape != mask.shape or values.shape != weights.shape:
        raise ValueError(
            "values, mask and weights should have same shape, "
            f"got {values.shape}, {mask.shape}, {weights.shape}"
        )

    weighted_mask = mask * weights.clamp_min(0.0)
    return (values * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def normal_kl_raw(
    mean_q: torch.Tensor,
    std_q: torch.Tensor,
    mean_p: torch.Tensor,
    std_p: torch.Tensor,
) -> torch.Tensor:
    """
    Diagonal Gaussian KL(q || p), returned as (B, T, 1).
    """
    q_var = std_q.pow(2)
    p_var = std_p.pow(2)

    kl = (
        torch.log(std_p)
        - torch.log(std_q)
        + 0.5 * (q_var + (mean_q - mean_p).pow(2)) / p_var
        - 0.5
    )

    return kl.sum(dim=-1, keepdim=True)


def kl_balancing_loss(
    posterior: RSSMState,
    prior: RSSMState,
    free_nats: float = 1.0,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
) -> torch.Tensor:
    """
    KL balancing used by Dreamer-like RSSM training.

    Dynamic loss:
        KL(stopgrad(posterior) || prior)

    Representation loss:
        KL(posterior || stopgrad(prior))
    """
    dyn_kl = normal_kl_raw(
        mean_q=posterior.mean.detach(),
        std_q=posterior.std.detach(),
        mean_p=prior.mean,
        std_p=prior.std,
    )

    rep_kl = normal_kl_raw(
        mean_q=posterior.mean,
        std_q=posterior.std,
        mean_p=prior.mean.detach(),
        std_p=prior.std.detach(),
    )

    if free_nats > 0.0:
        dyn_kl = torch.clamp(dyn_kl, min=free_nats)
        rep_kl = torch.clamp(rep_kl, min=free_nats)

    return dyn_scale * dyn_kl + rep_scale * rep_kl


def masked_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
) -> torch.Tensor:
    """
    Masked BCEWithLogits loss.
    """
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    weight = targets * positive_weight + (1.0 - targets) * negative_weight
    loss = loss * weight

    return masked_mean(loss, mask)


def detach_state(state: RSSMState) -> RSSMState:
    """
    Detach an RSSMState from the current computation graph.
    """
    return RSSMState(
        deter=state.deter.detach(),
        stoch=state.stoch.detach(),
        mean=state.mean.detach(),
        std=state.std.detach(),
    )


def set_requires_grad(modules: List[nn.Module], requires_grad: bool):
    """
    Temporarily enable / disable gradients for a group of modules.
    """
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(requires_grad)


def get_actor_feat_from_state(state: RSSMState) -> torch.Tensor:
    """
    Stable actor feature.

    The actor uses [deter, mean] instead of [deter, sampled stoch] to reduce
    action jitter during real-environment data collection.
    """
    return torch.cat([state.deter, state.mean], dim=-1)


def compute_world_model_loss(
    encoder: VisualEncoder,
    rssm: RSSM,
    decoder: VisualDecoder,
    reward_model: RewardModel,
    continue_model: ContinueModel,
    event_model: EventModel,
    obs_tensor: torch.Tensor,
    actions_tensor: torch.Tensor,
    reward_targets: torch.Tensor,
    done_targets: torch.Tensor,
    event_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    recon_scale: float = 1.0,
    reward_scale: float = 1.0,
    continue_scale: float = 1.0,
    event_scale: float = 1.0,
    kl_scale: float = 0.1,
    free_nats: float = 1.0,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    continue_terminal_weight: float = 8.0,
    prior_loss_scale: float = 0.5,
    success_reward_weight: float = 4.0,
    terminal_reward_weight: float = 2.0,
    event_positive_weights: Optional[torch.Tensor] = None,
) -> WorldModelLossOutput:
    """
    Compute world model loss without any expert-action supervision.

    Alignment:
        obs[:, t] corresponds to the state reached after actions[:, t].
        actions[:, 0] is a dummy previous action from reset.
    """
    embeddings = encode_sequence(
        encoder=encoder,
        obs_tensor=obs_tensor,
    )

    posteriors, priors = rssm.observe(
        embeddings=embeddings,
        actions=actions_tensor,
    )

    posterior_feat = rssm.get_feat(posteriors)
    prior_feat = rssm.get_feat(priors)

    # Reconstruction is trained from posterior states because the posterior has
    # access to the current observation embedding.
    reconstructed = decoder(posterior_feat)

    pred_rewards = reward_model(posterior_feat)
    pred_continue_logits = continue_model(posterior_feat)
    pred_event_logits = event_model(posterior_feat)

    # Prior head losses improve the reward / continue predictions actually used
    # during latent imagination.
    pred_prior_rewards = reward_model(prior_feat)
    pred_prior_continue_logits = continue_model(prior_feat)
    pred_prior_event_logits = event_model(prior_feat)

    recon_error = (reconstructed - obs_tensor).pow(2).mean(
        dim=(2, 3, 4),
        keepdim=False,
    ).unsqueeze(-1)

    posterior_reward_error = (pred_rewards - reward_targets).pow(2)
    prior_reward_error = (pred_prior_rewards - reward_targets).pow(2)

    reward_weights = torch.ones_like(reward_targets)
    reward_weights = reward_weights + success_reward_weight * (reward_targets > 0.0).float()
    reward_weights = reward_weights + terminal_reward_weight * done_targets.float()

    continue_targets = 1.0 - done_targets.float()

    posterior_continue_loss = masked_binary_loss(
        logits=pred_continue_logits,
        targets=continue_targets,
        mask=valid_mask,
        positive_weight=1.0,
        negative_weight=continue_terminal_weight,
    )

    prior_continue_loss = masked_binary_loss(
        logits=pred_prior_continue_logits,
        targets=continue_targets,
        mask=valid_mask,
        positive_weight=1.0,
        negative_weight=continue_terminal_weight,
    )

    if event_targets.shape[-1] != pred_event_logits.shape[-1]:
        raise ValueError(
            "event_targets last dim should match event_model output, "
            f"got {event_targets.shape[-1]} vs {pred_event_logits.shape[-1]}"
        )

    if event_positive_weights is None:
        event_positive_weights = torch.ones(
            pred_event_logits.shape[-1],
            device=pred_event_logits.device,
            dtype=pred_event_logits.dtype,
        )
    else:
        event_positive_weights = event_positive_weights.to(
            device=pred_event_logits.device,
            dtype=pred_event_logits.dtype,
        )

    event_pos_weight = event_positive_weights.view(
        *((1,) * (pred_event_logits.ndim - 1)),
        -1,
    )
    event_neg_weight = torch.ones_like(event_pos_weight)

    posterior_event_loss_map = F.binary_cross_entropy_with_logits(
        pred_event_logits,
        event_targets,
        reduction="none",
    )
    posterior_event_weight = event_targets * event_pos_weight + (1.0 - event_targets) * event_neg_weight
    posterior_event_loss = masked_mean(
        posterior_event_loss_map * posterior_event_weight,
        valid_mask.expand_as(posterior_event_loss_map),
    )

    prior_event_loss_map = F.binary_cross_entropy_with_logits(
        pred_prior_event_logits,
        event_targets,
        reduction="none",
    )
    prior_event_weight = event_targets * event_pos_weight + (1.0 - event_targets) * event_neg_weight
    prior_event_loss = masked_mean(
        prior_event_loss_map * prior_event_weight,
        valid_mask.expand_as(prior_event_loss_map),
    )

    recon_loss = masked_mean(recon_error, valid_mask)
    reward_loss = (
        masked_weighted_mean(posterior_reward_error, valid_mask, reward_weights)
        + prior_loss_scale * masked_weighted_mean(prior_reward_error, valid_mask, reward_weights)
    )
    continue_loss = posterior_continue_loss + prior_loss_scale * prior_continue_loss
    event_loss = posterior_event_loss + prior_loss_scale * prior_event_loss

    kl_error = kl_balancing_loss(
        posterior=posteriors,
        prior=priors,
        free_nats=free_nats,
        dyn_scale=dyn_scale,
        rep_scale=rep_scale,
    )
    kl_loss = masked_mean(kl_error, valid_mask)

    with torch.no_grad():
        continue_pred = (torch.sigmoid(pred_continue_logits) >= 0.5).float()
        continue_acc = (
            (continue_pred == continue_targets).float() * valid_mask
        ).sum() / valid_mask.sum().clamp_min(1.0)
        event_pred = (torch.sigmoid(pred_event_logits) >= 0.5).float()
        event_mask = valid_mask.expand_as(event_pred)
        event_acc = (
            (event_pred == event_targets).float() * event_mask
        ).sum() / event_mask.sum().clamp_min(1.0)

    total_loss = (
        recon_scale * recon_loss
        + reward_scale * reward_loss
        + continue_scale * continue_loss
        + event_scale * event_loss
        + kl_scale * kl_loss
    )

    return WorldModelLossOutput(
        total_loss=total_loss,
        recon_loss=recon_loss,
        reward_loss=reward_loss,
        continue_loss=continue_loss,
        event_loss=event_loss,
        kl_loss=kl_loss,
        continue_acc=continue_acc,
        event_acc=event_acc,
        reconstructed=reconstructed,
        pred_rewards=pred_rewards,
        pred_continue_logits=pred_continue_logits,
        pred_event_logits=pred_event_logits,
        posteriors=posteriors,
        priors=priors,
        posterior_feat=posterior_feat,
        prior_feat=prior_feat,
    )


def train_world_model_step(
    encoder: VisualEncoder,
    rssm: RSSM,
    decoder: VisualDecoder,
    reward_model: RewardModel,
    continue_model: ContinueModel,
    event_model: EventModel,
    optimizer: torch.optim.Optimizer,
    obs_tensor: torch.Tensor,
    actions_tensor: torch.Tensor,
    reward_targets: torch.Tensor,
    done_targets: torch.Tensor,
    event_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    recon_scale: float = 1.0,
    reward_scale: float = 1.0,
    continue_scale: float = 1.0,
    event_scale: float = 1.0,
    kl_scale: float = 0.1,
    free_nats: float = 1.0,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    continue_terminal_weight: float = 8.0,
    prior_loss_scale: float = 0.5,
    success_reward_weight: float = 4.0,
    terminal_reward_weight: float = 2.0,
    event_positive_weights: Optional[torch.Tensor] = None,
    grad_clip: float = 100.0,
) -> WorldModelLossOutput:
    """
    One world model update.
    """
    encoder.train()
    rssm.train()
    decoder.train()
    reward_model.train()
    continue_model.train()
    event_model.train()

    optimizer.zero_grad(set_to_none=True)

    loss_output = compute_world_model_loss(
        encoder=encoder,
        rssm=rssm,
        decoder=decoder,
        reward_model=reward_model,
        continue_model=continue_model,
        event_model=event_model,
        obs_tensor=obs_tensor,
        actions_tensor=actions_tensor,
        reward_targets=reward_targets,
        done_targets=done_targets,
        event_targets=event_targets,
        valid_mask=valid_mask,
        recon_scale=recon_scale,
        reward_scale=reward_scale,
        continue_scale=continue_scale,
        event_scale=event_scale,
        kl_scale=kl_scale,
        free_nats=free_nats,
        dyn_scale=dyn_scale,
        rep_scale=rep_scale,
        continue_terminal_weight=continue_terminal_weight,
        prior_loss_scale=prior_loss_scale,
        success_reward_weight=success_reward_weight,
        terminal_reward_weight=terminal_reward_weight,
        event_positive_weights=event_positive_weights,
    )

    loss_output.total_loss.backward()

    params = []
    for module in [encoder, rssm, decoder, reward_model, continue_model, event_model]:
        params.extend(list(module.parameters()))

    grad_norm = torch.nn.utils.clip_grad_norm_(
        params,
        max_norm=grad_clip,
    )

    optimizer.step()

    loss_output.grad_norm = float(grad_norm)

    return loss_output


def select_state_by_index(
    state: RSSMState,
    indices: torch.Tensor,
) -> RSSMState:
    """
    Select a different time index for each batch element from an RSSM state sequence.

    Args:
        state: fields shaped as (B, T, dim)
        indices: (B,)
    """
    batch_size = state.deter.shape[0]
    batch_indices = torch.arange(batch_size, device=state.deter.device)

    return RSSMState(
        deter=state.deter[batch_indices, indices],
        stoch=state.stoch[batch_indices, indices],
        mean=state.mean[batch_indices, indices],
        std=state.std[batch_indices, indices],
    )


def sample_imagination_init_indices(
    valid_mask: torch.Tensor,
    done_targets: torch.Tensor,
    collision_targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample non-terminal valid states as starting points for imagination.
    """
    valid = valid_mask.squeeze(-1) > 0.5
    done = done_targets.squeeze(-1) > 0.5
    nonterminal_valid = valid & (~done)
    if collision_targets is not None:
        collision = collision_targets.squeeze(-1) > 0.5
        nonterminal_valid = nonterminal_valid & (~collision)

    indices = []
    for b in range(valid.shape[0]):
        candidates = torch.nonzero(
            nonterminal_valid[b],
            as_tuple=False,
        ).squeeze(-1)

        if candidates.numel() == 0:
            candidates = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)

        if candidates.numel() == 0:
            indices.append(torch.tensor(0, device=valid.device, dtype=torch.long))
            continue

        pick = torch.randint(
            low=0,
            high=candidates.numel(),
            size=(1,),
            device=valid.device,
        )
        indices.append(candidates[pick].squeeze(0))

    return torch.stack(indices, dim=0)


def train_actor_bc_step(
    encoder: VisualEncoder,
    rssm: RSSM,
    actor: DiscreteActor,
    optimizer: torch.optim.Optimizer,
    obs_tensor: torch.Tensor,
    actions_tensor: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_scale: float = 1.0,
    grad_clip: float = 100.0,
) -> ActorBCLossOutput:
    encoder.eval()
    rssm.eval()
    actor.train()

    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        embeddings = encode_sequence(
            encoder=encoder,
            obs_tensor=obs_tensor,
        )
        posteriors, _ = rssm.observe(
            embeddings=embeddings,
            actions=actions_tensor,
        )
        # Actor should map current latent state s_t to the next real action a_{t+1}.
        # actions[:, t] is the action that produced obs[:, t], so BC must use a 1-step shift.
        bc_states = RSSMState(
            deter=posteriors.deter[:, :-1],
            stoch=posteriors.stoch[:, :-1],
            mean=posteriors.mean[:, :-1],
            std=posteriors.std[:, :-1],
        )
        actor_feat = get_actor_feat_from_state(bc_states)

    bc_targets = actions_tensor[:, 1:]
    bc_valid_mask = (valid_mask[:, :-1] * valid_mask[:, 1:]).contiguous()

    logits = actor.get_logits(actor_feat)
    log_probs = F.log_softmax(logits, dim=-1)
    per_step_loss = -log_probs.gather(dim=-1, index=bc_targets.unsqueeze(-1))
    bc_loss = masked_mean(per_step_loss, bc_valid_mask)

    scaled_bc_loss = loss_scale * bc_loss
    scaled_bc_loss.backward()

    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
        actor.parameters(),
        max_norm=grad_clip,
    )
    optimizer.step()

    with torch.no_grad():
        pred_actions = torch.argmax(logits, dim=-1)
        correct = (pred_actions == bc_targets).float().unsqueeze(-1)
        bc_acc = masked_mean(correct, bc_valid_mask)
        action_histogram = torch.bincount(
            bc_targets[bc_valid_mask.squeeze(-1) > 0.5].reshape(-1),
            minlength=actor.num_actions,
        ).float()
        action_histogram = action_histogram / action_histogram.sum().clamp_min(1.0)

    return ActorBCLossOutput(
        bc_loss=bc_loss.detach(),
        bc_acc=bc_acc.detach(),
        action_histogram=action_histogram.detach(),
        grad_norm_actor=float(actor_grad_norm),
    )


def train_actor_critic_step(
    rssm: RSSM,
    actor: DiscreteActor,
    reward_model: RewardModel,
    continue_model: ContinueModel,
    event_model: Optional[EventModel],
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    init_state: RSSMState,
    horizon: int = 8,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    entropy_scale: float = 1e-2,
    critic_scale: float = 1.0,
    continue_return_scale: float = 1.0,
    reward_clip: Optional[float] = None,
    success_bonus: float = 0.0,
    collision_penalty: float = 0.0,
    collision_continue_scale: float = 0.0,
    non_forward_penalty_scale: float = 0.0,
    forward_action_index: int = 0,
    real_feat: Optional[torch.Tensor] = None,
    real_rewards: Optional[torch.Tensor] = None,
    real_dones: Optional[torch.Tensor] = None,
    real_valid_mask: Optional[torch.Tensor] = None,
    real_critic_scale: float = 0.5,
    grad_clip: float = 100.0,
) -> ActorCriticLossOutput:
    """
    One imagination-based actor-critic update.

    This is the standard RL policy optimization path:
        latent state -> actor -> imagined trajectory -> return -> actor/critic update

    No expert action target is used.
    """
    if horizon <= 0:
        raise ValueError(f"horizon should be > 0, got {horizon}")

    actor.train()
    critic.train()

    # Do not update the world model during actor/critic optimization.
    frozen_modules = [rssm, reward_model, continue_model]
    if event_model is not None:
        frozen_modules.append(event_model)
    set_requires_grad(frozen_modules, False)

    init_state = detach_state(init_state)

    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)

    rollout = actor_imagination_rollout(
        rssm=rssm,
        actor=actor,
        reward_model=reward_model,
        continue_model=continue_model,
        critic=critic,
        init_state=init_state,
        horizon=horizon,
        gamma=gamma,
        lambda_=lambda_,
        deterministic=False,
        temperature=1.0,
    )

    rewards = rollout.rewards
    effective_continues = rollout.continues.detach()

    if event_model is not None:
        imagined_feat = rssm.get_feat(rollout.states)
        event_probs = torch.sigmoid(event_model(imagined_feat))
        success_prob = event_probs[..., 0:1]
        collision_prob = event_probs[..., 1:2]
        rewards = rewards + success_bonus * success_prob - collision_penalty * collision_prob
        effective_continues = effective_continues * (1.0 - collision_continue_scale * collision_prob)

    if non_forward_penalty_scale > 0.0:
        non_forward_mask = (rollout.actions != forward_action_index).float().unsqueeze(-1)
        rewards = rewards - non_forward_penalty_scale * non_forward_mask

    if reward_clip is not None and reward_clip > 0.0:
        rewards = rewards.clamp(-reward_clip, reward_clip)

    effective_continues = (
        effective_continues * continue_return_scale
    ).clamp(0.0, 1.0)

    returns = compute_lambda_returns(
        rewards=rewards,
        values=rollout.values,
        continues=effective_continues,
        bootstrap=rollout.terminal_value.detach(),
        gamma=gamma,
        lambda_=lambda_,
    )

    return_target = returns.detach()

    advantage = return_target - rollout.values.detach()
    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-5)
    advantage = advantage.clamp(-5.0, 5.0)

    policy_loss = -(
        rollout.log_probs.unsqueeze(-1) * advantage
    ).mean()

    entropy_loss = -rollout.entropies.mean()
    actor_loss = policy_loss + entropy_scale * entropy_loss

    imagined_critic_loss = F.mse_loss(rollout.values, return_target)

    real_critic_loss = torch.zeros(
        (),
        device=imagined_critic_loss.device,
        dtype=imagined_critic_loss.dtype,
    )

    if (
        real_feat is not None
        and real_rewards is not None
        and real_dones is not None
        and real_valid_mask is not None
        and real_feat.shape[1] >= 2
    ):
        current_feat = real_feat[:, :-1].detach()
        next_feat = real_feat[:, 1:].detach()
        transition_rewards = real_rewards[:, 1:].detach()
        transition_dones = real_dones[:, 1:].detach()
        transition_mask = (real_valid_mask[:, :-1] * real_valid_mask[:, 1:]).detach()

        current_values = critic(current_feat)
        with torch.no_grad():
            next_values = critic(next_feat)
            td_targets = transition_rewards + gamma * (1.0 - transition_dones) * next_values

        real_td_error = (current_values - td_targets).pow(2)
        real_critic_loss = masked_mean(real_td_error, transition_mask)

    critic_loss = imagined_critic_loss + real_critic_scale * real_critic_loss
    total_loss = actor_loss + critic_scale * critic_loss

    total_loss.backward()

    actor_grad_norm = torch.nn.utils.clip_grad_norm_(
        actor.parameters(),
        max_norm=grad_clip,
    )
    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
        critic.parameters(),
        max_norm=grad_clip,
    )

    actor_optimizer.step()
    critic_optimizer.step()

    set_requires_grad(frozen_modules, True)

    action_histogram = torch.bincount(
        rollout.actions.reshape(-1),
        minlength=actor.num_actions,
    ).float()
    action_histogram = action_histogram / action_histogram.sum().clamp_min(1.0)

    return ActorCriticLossOutput(
        total_loss=total_loss.detach(),
        actor_loss=actor_loss.detach(),
        critic_loss=critic_loss.detach(),
        imagined_critic_loss=imagined_critic_loss.detach(),
        real_critic_loss=real_critic_loss.detach(),
        entropy_loss=entropy_loss.detach(),
        mean_return=return_target.mean().detach(),
        mean_reward=rewards.mean().detach(),
        mean_value=rollout.values.mean().detach(),
        mean_continue=rollout.continues.mean().detach(),
        action_histogram=action_histogram.detach(),
        grad_norm_actor=float(actor_grad_norm),
        grad_norm_critic=float(critic_grad_norm),
    )
