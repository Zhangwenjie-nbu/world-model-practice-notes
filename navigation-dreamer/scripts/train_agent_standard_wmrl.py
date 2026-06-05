# scripts/train_agent_standard_wmrl.py

"""
Standard World Model RL training script.

This script removes expert BC / DAgger / heuristic-label training from the
training path. The policy is learned only through model-based actor-critic:

    random warmup data
    -> train world model
    -> imagine latent rollouts
    -> update actor / critic from imagined returns
    -> collect new real-environment data with the current actor
    -> repeat

Heuristic/BFS expert is intentionally not imported or used here.
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from envs.grid_nav_env import GridNavEnv
from memory.replay_buffer import EpisodeReplayBuffer
from models.actor import DiscreteActor
from models.continue_model import ContinueModel
from models.critic import Critic
from models.decoder import VisualDecoder
from models.encoder import VisualEncoder
from models.event_model import EventModel
from models.reward_model import RewardModel
from models.rssm import RSSM
from tools.seed import set_global_seed
from trainers.joint_trainer_standard import (
    compute_world_model_loss,
    get_actor_feat_from_state,
    sample_imagination_init_indices,
    select_state_by_index,
    train_actor_bc_step,
    train_actor_critic_step,
    train_world_model_step,
)


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert one uint8 HWC observation into BCHW float tensor.
    """
    x = torch.from_numpy(obs).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x.to(device)


def batch_to_tensors(batch: Dict[str, np.ndarray], device: torch.device):
    """
    Convert replay batch to torch tensors.

    No expert action or event target is returned in the standard RL version.
    """
    obs = batch["obs"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    dones = batch["dones"]
    valid = batch["valid"]
    successes = batch["successes"]
    collisions = batch["collisions"]

    # Keep observations as uint8 on CPU and convert on GPU to reduce transfer volume.
    obs_tensor = torch.from_numpy(obs).to(device=device, non_blocking=True)
    obs_tensor = obs_tensor.permute(0, 1, 4, 2, 3).contiguous()
    obs_tensor = obs_tensor.float().div_(255.0)

    actions_tensor = torch.from_numpy(actions).to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )
    reward_targets = torch.from_numpy(rewards).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).unsqueeze(-1)
    done_targets = torch.from_numpy(dones.astype(np.float32)).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).unsqueeze(-1)
    valid_mask = torch.from_numpy(valid).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).unsqueeze(-1)
    success_targets = torch.from_numpy(successes.astype(np.float32)).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).unsqueeze(-1)
    collision_targets = torch.from_numpy(collisions.astype(np.float32)).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    ).unsqueeze(-1)
    event_targets = torch.cat([success_targets, collision_targets], dim=-1)

    return obs_tensor, actions_tensor, reward_targets, done_targets, event_targets, valid_mask


def add_episode_to_buffer(buffer: EpisodeReplayBuffer, episode: dict):
    """
    Add one episode to replay buffer.
    """
    buffer.add_episode(
        obs=episode["obs"],
        actions=episode["actions"],
        rewards=episode["rewards"],
        dones=episode["dones"],
        successes=episode.get("successes", None),
        collisions=episode.get("collisions", None),
        start_path_length=episode.get("start_path_length", None),
    )


def _collect_episode_generic(env: GridNavEnv, max_steps: int, action_fn):
    """
    Generic episode collector.

    This function does not use any expert. action_fn may be random policy or
    current actor policy.
    """
    obs_list = []
    action_list = []
    reward_list = []
    done_list = []
    success_list = []
    collision_list = []

    obs = env.reset()
    start_path_length = env.get_current_shortest_path_length()

    obs_list.append(obs)
    action_list.append(GridNavEnv.ACTION_FORWARD)
    reward_list.append(0.0)
    done_list.append(False)
    success_list.append(False)
    collision_list.append(False)

    total_reward = 0.0
    success = False
    collision = False

    for _ in range(max_steps):
        action = int(action_fn(env))

        next_obs, reward, done, info = env.step(action)

        obs_list.append(next_obs)
        action_list.append(action)
        reward_list.append(float(reward))
        done_list.append(bool(done))

        total_reward += float(reward)
        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))
        success_list.append(success)
        collision_list.append(collision)

        obs = next_obs

        if done:
            break

    return {
        "obs": np.stack(obs_list, axis=0),
        "actions": np.array(action_list, dtype=np.int64),
        "rewards": np.array(reward_list, dtype=np.float32),
        "dones": np.array(done_list, dtype=np.bool_),
        "total_reward": total_reward,
        "success": success,
        "collision": collision,
        "length": len(obs_list),
        "start_path_length": start_path_length,
        "successes": np.array(success_list, dtype=np.bool_),
        "collisions": np.array(collision_list, dtype=np.bool_),
    }


@torch.no_grad()
def collect_episode_random(
    env: GridNavEnv,
    max_steps: int,
    action_probs=None,
):
    """
    Collect one episode with non-expert exploratory random actions.

    This is intentionally not uniform random by default. In this navigation
    environment, uniform random produces too many immediate collision episodes
    and too few useful transition sequences. The default distribution biases
    exploration toward moving and turning;

    This policy does not use BFS, shortest paths, expert labels, or goal-state
    privileged planning; it is only an action-prior for exploration.
    """
    if action_probs is None:
        action_probs = np.array([0.4, 0.3, 0.3], dtype=np.float64)
    else:
        action_probs = np.asarray(action_probs, dtype=np.float64)

    if action_probs.shape != (env.num_actions,):
        raise ValueError(
            f"action_probs should have shape ({env.num_actions},), "
            f"but got {action_probs.shape}"
        )

    prob_sum = float(action_probs.sum())
    if prob_sum <= 0.0:
        raise ValueError("action_probs must have positive sum.")

    action_probs = action_probs / prob_sum
    actions = np.arange(env.num_actions)

    return _collect_episode_generic(
        env=env,
        max_steps=max_steps,
        action_fn=lambda e: int(np.random.choice(actions, p=action_probs)),
    )


@torch.no_grad()
def collect_episode_actor_policy(
    env: GridNavEnv,
    encoder: VisualEncoder,
    rssm: RSSM,
    actor: DiscreteActor,
    device: torch.device,
    max_steps: int,
    epsilon: float = 0.1,
    temperature: float = 1.0,
    deterministic: bool = False,
    exploration_action_probs=None,
):
    """
    Collect one real-environment episode with the current actor.

    Exploration:
        with probability epsilon, choose an exploratory random action sampled from
        [forward, left, right] = [0.50, 0.25, 0.25];
        otherwise sample from the actor distribution.

    No expert action is used.
    """
    if exploration_action_probs is None:
        exploration_action_probs = np.array([0.50, 0.25, 0.25], dtype=np.float64)
    else:
        exploration_action_probs = np.asarray(exploration_action_probs, dtype=np.float64)

    if exploration_action_probs.shape != (env.num_actions,):
        raise ValueError(
            f"exploration_action_probs should have shape ({env.num_actions},), "
            f"but got {exploration_action_probs.shape}"
        )

    prob_sum = float(exploration_action_probs.sum())
    if prob_sum <= 0.0:
        raise ValueError("exploration_action_probs must have positive sum.")

    exploration_action_probs = exploration_action_probs / prob_sum

    encoder.eval()
    rssm.eval()
    actor.eval()

    obs_list = []
    action_list = []
    reward_list = []
    done_list = []
    success_list = []
    collision_list = []

    obs = env.reset()
    start_path_length = env.get_current_shortest_path_length()
    state = rssm.initial_state(batch_size=1, device=device)
    prev_action = GridNavEnv.ACTION_FORWARD

    obs_list.append(obs)
    action_list.append(prev_action)
    reward_list.append(0.0)
    done_list.append(False)
    success_list.append(False)
    collision_list.append(False)

    total_reward = 0.0
    success = False
    collision = False

    for _ in range(max_steps):
        obs_tensor = obs_to_tensor(obs, device=device)
        embedding = encoder(obs_tensor)

        prev_action_tensor = torch.tensor(
            [prev_action],
            dtype=torch.long,
            device=device,
        )

        state, _ = rssm.obs_step(
            prev_state=state,
            action=prev_action_tensor,
            embedding=embedding,
        )

        feat = get_actor_feat_from_state(state)

        if np.random.rand() < epsilon:
            action = int(
                np.random.choice(
                    np.arange(env.num_actions),
                    p=exploration_action_probs,
                )
            )
        else:
            action_tensor, _, _ = actor.sample_action(
                feat=feat,
                deterministic=deterministic,
                temperature=temperature,
            )
            action = int(action_tensor.item())

        next_obs, reward, done, info = env.step(action)

        obs_list.append(next_obs)
        action_list.append(action)
        reward_list.append(float(reward))
        done_list.append(bool(done))

        total_reward += float(reward)
        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))
        success_list.append(success)
        collision_list.append(collision)

        prev_action = action
        obs = next_obs

        if done:
            break

    return {
        "obs": np.stack(obs_list, axis=0),
        "actions": np.array(action_list, dtype=np.int64),
        "rewards": np.array(reward_list, dtype=np.float32),
        "dones": np.array(done_list, dtype=np.bool_),
        "total_reward": total_reward,
        "success": success,
        "collision": collision,
        "length": len(obs_list),
        "start_path_length": start_path_length,
        "successes": np.array(success_list, dtype=np.bool_),
        "collisions": np.array(collision_list, dtype=np.bool_),
    }


@torch.no_grad()
def evaluate_actor_agent(
    env: GridNavEnv,
    encoder: VisualEncoder,
    rssm: RSSM,
    actor: DiscreteActor,
    device: torch.device,
    num_episodes: int,
    max_steps: int,
    deterministic: bool = True,
    temperature: float = 1.0,
):
    """
    Evaluate current actor in the real environment.
    """
    encoder.eval()
    rssm.eval()
    actor.eval()

    rewards = []
    successes = []
    collisions = []
    lengths = []
    action_counts = np.zeros(env.num_actions, dtype=np.int64)
    prob_sum = np.zeros(env.num_actions, dtype=np.float64)
    prob_count = 0

    for _ in range(num_episodes):
        obs = env.reset()
        state = rssm.initial_state(batch_size=1, device=device)
        prev_action = GridNavEnv.ACTION_FORWARD

        total_reward = 0.0
        success = False
        collision = False
        length = 0

        for _ in range(max_steps):
            obs_tensor = obs_to_tensor(obs, device=device)
            embedding = encoder(obs_tensor)

            prev_action_tensor = torch.tensor(
                [prev_action],
                dtype=torch.long,
                device=device,
            )
            state, _ = rssm.obs_step(
                prev_state=state,
                action=prev_action_tensor,
                embedding=embedding,
            )

            feat = get_actor_feat_from_state(state)
            dist = actor(feat, temperature=temperature)
            probs = dist.probs[0].detach().cpu().numpy()
            prob_sum += probs
            prob_count += 1

            if deterministic:
                action = int(np.argmax(probs))
            else:
                action = int(dist.sample().item())

            action_counts[action] += 1

            next_obs, reward, done, info = env.step(action)

            total_reward += float(reward)
            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            length += 1

            prev_action = action
            obs = next_obs

            if done:
                break

        rewards.append(total_reward)
        successes.append(float(success))
        collisions.append(float(collision))
        lengths.append(float(length))

    avg_probs = prob_sum / max(prob_count, 1)

    return {
        "avg_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "avg_length": float(np.mean(lengths)),
        "action_counts": action_counts.tolist(),
        "avg_probs": [float(x) for x in avg_probs],
    }


@torch.no_grad()
def evaluate_random_policy(
    env: GridNavEnv,
    num_episodes: int,
    max_steps: int,
    action_probs=None,
):
    """
    Exploration random policy baseline. No expert is used.
    """
    results = []
    for _ in range(num_episodes):
        ep = collect_episode_random(
            env=env,
            max_steps=max_steps,
            action_probs=action_probs,
        )
        results.append(ep)

    return {
        "avg_reward": float(np.mean([ep["total_reward"] for ep in results])),
        "success_rate": float(np.mean([float(ep["success"]) for ep in results])),
        "collision_rate": float(np.mean([float(ep["collision"]) for ep in results])),
        "avg_length": float(np.mean([float(ep["length"]) for ep in results])),
    }


def save_checkpoint(
    save_path: Path,
    encoder: VisualEncoder,
    rssm: RSSM,
    decoder: VisualDecoder,
    reward_model: RewardModel,
    continue_model: ContinueModel,
    event_model: EventModel,
    actor: DiscreteActor,
    critic: Critic,
    step: int,
    best_success: Optional[float] = None,
):
    """
    Save standard world-model RL checkpoint.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": step,
        "encoder": encoder.state_dict(),
        "rssm": rssm.state_dict(),
        "decoder": decoder.state_dict(),
        "reward_model": reward_model.state_dict(),
        "continue_model": continue_model.state_dict(),
        "event_model": event_model.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "standard_wmrl": True,
    }

    if best_success is not None:
        payload["best_success"] = float(best_success)

    torch.save(payload, save_path)


def linear_schedule(
    step: int,
    start: float,
    end: float,
    duration: int,
) -> float:
    """
    Linear decay schedule.
    """
    if duration <= 0:
        return end

    mix = min(max(step / float(duration), 0.0), 1.0)
    return float(start + mix * (end - start))


def configure_reset_curriculum(
    env: GridNavEnv,
    step: int,
    start_step: int,
    duration: int,
    min_path_length: int,
    start_max_path_length: int,
    end_max_path_length: int,
    hold_end_steps: int = 0,
    unlocked_max_path_length: Optional[int] = None,
) -> str:
    """
    Curriculum on reset difficulty measured by shortest-path length.

    Early training samples easier start-goal pairs. A short hold phase keeps
    the final constrained difficulty before reset falls back to full random.
    The time-based target difficulty is additionally capped by
    unlocked_max_path_length so progression can be gated by evaluation success.
    """
    if duration <= 0 or step < start_step:
        env.clear_reset_path_length_range()
        return "full_random"

    curriculum_step = step - start_step
    bounded_step = min(max(curriculum_step, 0), duration)
    if bounded_step >= duration:
        if hold_end_steps > 0 and curriculum_step < duration + hold_end_steps:
            final_max = end_max_path_length
            if unlocked_max_path_length is not None:
                final_max = min(final_max, unlocked_max_path_length)

            env.set_reset_path_length_range(
                min_path_length=min_path_length,
                max_path_length=final_max,
            )
            suffix = "_hold" if final_max >= end_max_path_length else "_gated"
            return f"path_len=[{min_path_length},{final_max}]{suffix}"

        env.clear_reset_path_length_range()
        return "full_random"

    current_max_path_length = int(
        round(
            linear_schedule(
                step=bounded_step,
                start=float(start_max_path_length),
                end=float(end_max_path_length),
                duration=duration,
            )
        )
    )
    current_max_path_length = max(current_max_path_length, min_path_length)

    if unlocked_max_path_length is not None:
        current_max_path_length = min(current_max_path_length, unlocked_max_path_length)

    env.set_reset_path_length_range(
        min_path_length=min_path_length,
        max_path_length=current_max_path_length,
    )
    return f"path_len=[{min_path_length},{current_max_path_length}]"


def get_distance_stage(max_path_length: Optional[int], base_threshold: int) -> int:
    if max_path_length is None or base_threshold <= 0:
        return 0

    if max_path_length >= base_threshold + 4:
        return 3

    if max_path_length >= base_threshold + 2:
        return 2

    if max_path_length >= base_threshold:
        return 1

    return 0


def get_full_random_adaptation_progress(
    step: int,
    start_step: int,
    duration: int,
    hold_end_steps: int,
    adaptation_steps: int,
) -> Optional[int]:
    if adaptation_steps <= 0:
        return None

    full_random_start_step = start_step + duration + hold_end_steps
    if step < full_random_start_step:
        return None

    adaptation_step = step - full_random_start_step
    if adaptation_step >= adaptation_steps:
        return None

    return adaptation_step


def is_hard_distance_phase(
    env: GridNavEnv,
    path_length_threshold: int,
) -> bool:
    return get_distance_stage(env.reset_max_path_length, path_length_threshold) > 0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--total-train-steps", type=int, default=100000)
    parser.add_argument("--actor-start-step", type=int, default=20000)
    parser.add_argument("--initial-random-episodes", type=int, default=500)
    parser.add_argument("--replay-capacity", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--collect-interval", type=int, default=5)
    parser.add_argument("--model-updates-per-step", type=int, default=1)
    parser.add_argument("--actor-updates-per-step", type=int, default=1)
    parser.add_argument("--actor-ac-interval", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument(
        "--terminate-on-collision",
        action="store_true",
        default=True,
        help=(
            "If set, collision immediately terminates the episode. "
            "By default standard WM-RL training keeps the episode alive after "
            "collision and only applies collision penalties."
        ),
    )
    parser.add_argument("--random-forward-prob", type=float, default=0.40)
    parser.add_argument("--random-left-prob", type=float, default=0.30)
    parser.add_argument("--random-right-prob", type=float, default=0.30)
    parser.add_argument("--reset-curriculum-start-step", type=int, default=None)
    parser.add_argument("--reset-curriculum-duration", type=int, default=40000)
    parser.add_argument("--curriculum-min-path-length", type=int, default=1)
    parser.add_argument("--curriculum-start-max-path-length", type=int, default=4)
    parser.add_argument("--curriculum-end-max-path-length", type=int, default=12)
    parser.add_argument("--curriculum-hold-end-steps", type=int, default=10000)
    parser.add_argument("--full-random-adaptation-steps", type=int, default=15000)
    parser.add_argument("--full-random-epsilon-boost", type=float, default=0.20)
    parser.add_argument("--full-random-temperature-boost", type=float, default=0.30)
    parser.add_argument("--full-random-adaptation-horizon", type=int, default=10)
    parser.add_argument("--full-random-adaptation-entropy-scale", type=float, default=3e-2)
    parser.add_argument("--hard-path-length-threshold", type=int, default=7)
    parser.add_argument("--hard-path-epsilon-boost", type=float, default=0.10)
    parser.add_argument("--hard-path-temperature-boost", type=float, default=0.20)
    parser.add_argument("--hard-path-horizon", type=int, default=8)
    parser.add_argument("--hard-path-entropy-scale", type=float, default=3e-2)
    parser.add_argument("--hard-path-real-critic-scale", type=float, default=1.0)
    parser.add_argument("--hard-path-stage2-epsilon-boost", type=float, default=0.18)
    parser.add_argument("--hard-path-stage2-temperature-boost", type=float, default=0.35)
    parser.add_argument("--hard-path-stage2-horizon", type=int, default=6)
    parser.add_argument("--hard-path-stage2-entropy-scale", type=float, default=4e-2)
    parser.add_argument("--hard-path-stage2-real-critic-scale", type=float, default=1.5)
    parser.add_argument("--hard-path-stage3-epsilon-boost", type=float, default=0.25)
    parser.add_argument("--hard-path-stage3-temperature-boost", type=float, default=0.50)
    parser.add_argument("--hard-path-stage3-horizon", type=int, default=5)
    parser.add_argument("--hard-path-stage3-entropy-scale", type=float, default=5e-2)
    parser.add_argument("--hard-path-stage3-real-critic-scale", type=float, default=2.0)
    parser.add_argument("--curriculum-advance-success-threshold", type=float, default=0.85)
    parser.add_argument("--curriculum-advance-min-evals", type=int, default=2)
    parser.add_argument("--goal-reward", type=float, default=20.0)
    parser.add_argument("--collision-penalty", type=float, default=8.0)
    parser.add_argument("--turn-penalty", type=float, default=0.05)
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--step-penalty", type=float, default=0.1)
    parser.add_argument("--success-buffer-capacity", type=int, default=400)
    parser.add_argument("--success-sample-ratio", type=float, default=0.35)
    parser.add_argument("--success-terminal-sample-ratio", type=float, default=0.8)
    parser.add_argument("--success-prioritize-long-paths", action="store_true", default=True)
    parser.add_argument("--success-long-path-sampling-alpha", type=float, default=1.0)
    parser.add_argument("--success-reward-weight", type=float, default=4.0)
    parser.add_argument("--terminal-reward-weight", type=float, default=2.0)
    parser.add_argument("--real-critic-scale", type=float, default=0.5)
    parser.add_argument("--ac-non-forward-penalty", type=float, default=0.0)
    parser.add_argument("--ac-collision-continue-scale", type=float, default=0.90)
    parser.add_argument("--actor-bc-scale", type=float, default=0.25)
    parser.add_argument("--actor-bc-updates-per-step", type=int, default=1)
    parser.add_argument("--collapse-restore-threshold", type=float, default=0.20)
    parser.add_argument("--collapse-restore-min-best-success", type=float, default=0.70)
    parser.add_argument("--collapse-restore-collision-threshold", type=float, default=0.30)
    parser.add_argument("--collapse-ac-pause-steps", type=int, default=3000)
    parser.add_argument("--collapse-actor-lr-decay", type=float, default=0.5)
    parser.add_argument("--collapse-bc-boost", type=float, default=2.0)
    parser.add_argument("--imagination-horizon", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    random_action_probs = np.array(
        [
            args.random_forward_prob,
            args.random_left_prob,
            args.random_right_prob,
        ],
        dtype=np.float64,
    )
    if random_action_probs.sum() <= 0.0:
        raise ValueError("Random action probabilities must have positive sum.")
    random_action_probs = random_action_probs / random_action_probs.sum()

    if args.curriculum_min_path_length < 1:
        raise ValueError("curriculum_min_path_length should be >= 1.")

    if args.curriculum_start_max_path_length < args.curriculum_min_path_length:
        raise ValueError(
            "curriculum_start_max_path_length should be >= curriculum_min_path_length."
        )

    if args.curriculum_end_max_path_length < args.curriculum_min_path_length:
        raise ValueError(
            "curriculum_end_max_path_length should be >= curriculum_min_path_length."
        )

    if args.reset_curriculum_start_step is None:
        args.reset_curriculum_start_step = args.actor_start_step

    if args.reset_curriculum_start_step < 0:
        raise ValueError("reset_curriculum_start_step should be >= 0.")

    if args.curriculum_hold_end_steps < 0:
        raise ValueError("curriculum_hold_end_steps should be >= 0.")

    if args.full_random_adaptation_steps < 0:
        raise ValueError("full_random_adaptation_steps should be >= 0.")

    if args.full_random_adaptation_horizon <= 0:
        raise ValueError("full_random_adaptation_horizon should be > 0.")

    if args.hard_path_length_threshold < 0:
        raise ValueError("hard_path_length_threshold should be >= 0.")

    if args.hard_path_horizon <= 0:
        raise ValueError("hard_path_horizon should be > 0.")

    if args.hard_path_stage2_horizon <= 0:
        raise ValueError("hard_path_stage2_horizon should be > 0.")

    if args.hard_path_stage3_horizon <= 0:
        raise ValueError("hard_path_stage3_horizon should be > 0.")

    if not 0.0 <= args.curriculum_advance_success_threshold <= 1.0:
        raise ValueError("curriculum_advance_success_threshold should be in [0, 1].")

    if args.curriculum_advance_min_evals <= 0:
        raise ValueError("curriculum_advance_min_evals should be > 0.")

    if args.progress_reward_scale < 0.0:
        raise ValueError("progress_reward_scale should be non-negative.")

    if args.step_penalty < 0.0:
        raise ValueError("step_penalty should be non-negative.")

    if args.success_buffer_capacity >= args.replay_capacity:
        raise ValueError("success_buffer_capacity should be smaller than replay_capacity.")

    if not 0.0 <= args.success_terminal_sample_ratio <= 1.0:
        raise ValueError("success_terminal_sample_ratio should be in [0, 1].")

    if args.success_long_path_sampling_alpha < 0.0:
        raise ValueError("success_long_path_sampling_alpha should be non-negative.")

    if args.success_reward_weight < 0.0:
        raise ValueError("success_reward_weight should be non-negative.")

    if args.terminal_reward_weight < 0.0:
        raise ValueError("terminal_reward_weight should be non-negative.")

    if args.real_critic_scale < 0.0:
        raise ValueError("real_critic_scale should be non-negative.")

    if args.hard_path_real_critic_scale < 0.0:
        raise ValueError("hard_path_real_critic_scale should be non-negative.")

    if args.hard_path_stage2_real_critic_scale < 0.0:
        raise ValueError("hard_path_stage2_real_critic_scale should be non-negative.")

    if args.hard_path_stage3_real_critic_scale < 0.0:
        raise ValueError("hard_path_stage3_real_critic_scale should be non-negative.")

    set_global_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("Standard World Model RL training")
    print(f"device={device}")
    print("No expert BC / no DAgger / no heuristic labels in training. Action space: forward / turn_left / turn_right.")
    print(f"terminate_on_collision={args.terminate_on_collision}")
    print(f"exploration_random_action_probs={random_action_probs.tolist()}")
    print(
        "reset_curriculum="
        f"start_step={args.reset_curriculum_start_step}, "
        f"duration={args.reset_curriculum_duration}, "
        f"hold_end_steps={args.curriculum_hold_end_steps}, "
        f"path_len=[{args.curriculum_min_path_length}, "
        f"{args.curriculum_start_max_path_length} -> {args.curriculum_end_max_path_length}] then full_random"
    )
    print(
        f"success_replay=capacity={args.success_buffer_capacity}, "
        f"sample_ratio={args.success_sample_ratio:.2f}, "
        f"terminal_window_ratio={args.success_terminal_sample_ratio:.2f}, "
        f"prioritize_long_paths={args.success_prioritize_long_paths}, "
        f"long_path_alpha={args.success_long_path_sampling_alpha:.2f}"
    )
    print(
        f"reward_weighting=success+{args.success_reward_weight:.1f}, "
        f"terminal+{args.terminal_reward_weight:.1f}; "
        f"goal={args.goal_reward:.1f}, collision={args.collision_penalty:.1f}, "
        f"turn={args.turn_penalty:.2f}, progress={args.progress_reward_scale:.2f}, step={args.step_penalty:.2f}; "
        f"real_critic_scale={args.real_critic_scale:.2f}; "
        f"ac_non_forward_penalty={args.ac_non_forward_penalty:.2f}, "
        f"ac_collision_continue_scale={args.ac_collision_continue_scale:.2f}, "
        f"actor_ac_interval={args.actor_ac_interval}, "
        f"actor_bc_scale={args.actor_bc_scale:.2f}, actor_bc_updates={args.actor_bc_updates_per_step}; "
        f"imagination_horizon={args.imagination_horizon}"
    )
    print(
        f"hard_distance_phase=threshold={args.hard_path_length_threshold}, "
        f"stage1(eps={args.hard_path_epsilon_boost:.2f}, temp={args.hard_path_temperature_boost:.2f}, h={args.hard_path_horizon}, ent={args.hard_path_entropy_scale:.3f}, rc={args.hard_path_real_critic_scale:.2f}), "
        f"stage2(eps={args.hard_path_stage2_epsilon_boost:.2f}, temp={args.hard_path_stage2_temperature_boost:.2f}, h={args.hard_path_stage2_horizon}, ent={args.hard_path_stage2_entropy_scale:.3f}, rc={args.hard_path_stage2_real_critic_scale:.2f}), "
        f"stage3(eps={args.hard_path_stage3_epsilon_boost:.2f}, temp={args.hard_path_stage3_temperature_boost:.2f}, h={args.hard_path_stage3_horizon}, ent={args.hard_path_stage3_entropy_scale:.3f}, rc={args.hard_path_stage3_real_critic_scale:.2f})"
    )
    print(
        f"curriculum_gate=success_threshold={args.curriculum_advance_success_threshold:.2f}, "
        f"min_evals={args.curriculum_advance_min_evals}"
    )
    print("=" * 80)

    train_env = GridNavEnv(
        map_width=10,
        map_height=10,
        image_size=64,
        max_steps=args.max_episode_steps,
        goal_reward=args.goal_reward,
        collision_penalty=args.collision_penalty,
        turn_penalty=args.turn_penalty,
        progress_reward_scale=args.progress_reward_scale,
        step_penalty=args.step_penalty,
        random_reset=True,
        terminate_on_collision=args.terminate_on_collision,
        seed=42,
    )

    eval_env = GridNavEnv(
        map_width=10,
        map_height=10,
        image_size=64,
        max_steps=args.max_episode_steps,
        goal_reward=args.goal_reward,
        collision_penalty=args.collision_penalty,
        turn_penalty=args.turn_penalty,
        progress_reward_scale=args.progress_reward_scale,
        step_penalty=args.step_penalty,
        random_reset=True,
        terminate_on_collision=args.terminate_on_collision,
        seed=999,
    )

    random_eval_env = GridNavEnv(
        map_width=10,
        map_height=10,
        image_size=64,
        max_steps=args.max_episode_steps,
        goal_reward=args.goal_reward,
        collision_penalty=args.collision_penalty,
        turn_penalty=args.turn_penalty,
        progress_reward_scale=args.progress_reward_scale,
        step_penalty=args.step_penalty,
        random_reset=True,
        terminate_on_collision=args.terminate_on_collision,
        seed=2026,
    )

    encoder = VisualEncoder(image_size=64, in_channels=3, embedding_dim=256).to(device)
    rssm = RSSM(
        embedding_dim=256,
        num_actions=3,
        action_embed_dim=32,
        deter_dim=256,
        stoch_dim=32,
        hidden_dim=256,
        min_std=0.1,
    ).to(device)
    decoder = VisualDecoder(
        feature_dim=288,
        image_size=64,
        out_channels=3,
        hidden_channels=256,
    ).to(device)
    reward_model = RewardModel(feature_dim=288, hidden_dim=256).to(device)
    continue_model = ContinueModel(feature_dim=288, hidden_dim=256).to(device)
    event_model = EventModel(feature_dim=288, hidden_dim=256, num_events=2).to(device)
    actor = DiscreteActor(feature_dim=288, hidden_dim=256, num_actions=3).to(device)
    critic = Critic(feature_dim=288, hidden_dim=256).to(device)

    world_model_optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(rssm.parameters())
        + list(decoder.parameters())
        + list(reward_model.parameters())
        + list(continue_model.parameters())
        + list(event_model.parameters()),
        lr=3e-4,
    )

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-6)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)

    replay_buffer = EpisodeReplayBuffer(
        capacity_episodes=args.replay_capacity,
        padding_action=GridNavEnv.ACTION_FORWARD,
        success_capacity_episodes=args.success_buffer_capacity,
        success_sample_ratio=args.success_sample_ratio,
        success_terminal_sample_ratio=args.success_terminal_sample_ratio,
        prioritize_long_success=args.success_prioritize_long_paths,
        long_success_sampling_alpha=args.success_long_path_sampling_alpha,
    )

    print("\n[Random policy baseline]")
    random_metrics = evaluate_random_policy(
        env=random_eval_env,
        num_episodes=50,
        max_steps=args.max_episode_steps,
        action_probs=random_action_probs,
    )
    print(
        f"avg_reward={random_metrics['avg_reward']:.3f}, "
        f"success={random_metrics['success_rate']:.3f}, "
        f"collision={random_metrics['collision_rate']:.3f}, "
        f"avg_length={random_metrics['avg_length']:.1f}"
    )

    print("\n[Initial random replay collection]")
    for i in range(args.initial_random_episodes):
        train_env.clear_reset_path_length_range()
        reset_mode = "full_random"

        episode = collect_episode_random(
            env=train_env,
            max_steps=args.max_episode_steps,
            action_probs=random_action_probs,
        )
        add_episode_to_buffer(replay_buffer, episode)

        if (i + 1) % 50 == 0:
            print(
                f"random_ep={i + 1}/{args.initial_random_episodes}, "
                f"reset={reset_mode}, "
                f"buffer_ep={len(replay_buffer)}, buffer_steps={replay_buffer.num_steps}, "
                f"success_ep={replay_buffer.num_success_episodes}, "
                f"last_reward={episode['total_reward']:.3f}, "
                f"success={episode['success']}, collision={episode['collision']}, "
                f"length={episode['length']}"
            )

    print(
        f"\nReplay buffer ready: episodes={len(replay_buffer)}, "
        f"steps={replay_buffer.num_steps}"
    )

    best_success = -1.0
    best_actor_state = None
    best_critic_state = None
    collapse_ac_pause_until = -1
    collapse_bc_scale_multiplier = 1.0
    last_collect_mode = "initial_random"
    reset_mode = "full_random"
    wm_loss = None
    ac_loss = None
    bc_loss = None
    unlocked_max_path_length = args.curriculum_start_max_path_length
    curriculum_success_streak = 0

    for step in range(1, args.total_train_steps + 1):
        reset_mode = configure_reset_curriculum(
            env=train_env,
            step=step,
            start_step=args.reset_curriculum_start_step,
            duration=args.reset_curriculum_duration,
            min_path_length=args.curriculum_min_path_length,
            start_max_path_length=args.curriculum_start_max_path_length,
            end_max_path_length=args.curriculum_end_max_path_length,
            hold_end_steps=args.curriculum_hold_end_steps,
            unlocked_max_path_length=unlocked_max_path_length,
        )

        # Real-environment data collection.
        # Important: before actor_start_step the actor is still untrained, so
        # it must NOT be used to collect data. Otherwise replay is polluted by
        # untrained turn trajectories.
        if step % args.collect_interval == 0:
            if step < args.actor_start_step:
                episode = collect_episode_random(
                    env=train_env,
                    max_steps=args.max_episode_steps,
                    action_probs=random_action_probs,
                )
                last_collect_mode = "random_before_actor_start"
            else:
                # Schedule starts at actor_start_step instead of global step so
                # the first actor-collected episodes still have strong exploration.
                actor_phase_step = step - args.actor_start_step
                epsilon = linear_schedule(
                    step=actor_phase_step,
                    start=0.50,
                    end=0.05,
                    duration=50000,
                )
                temperature = linear_schedule(
                    step=actor_phase_step,
                    start=1.80,
                    end=0.90,
                    duration=50000,
                )
                full_random_adapt_step = get_full_random_adaptation_progress(
                    step=step,
                    start_step=args.reset_curriculum_start_step,
                    duration=args.reset_curriculum_duration,
                    hold_end_steps=args.curriculum_hold_end_steps,
                    adaptation_steps=args.full_random_adaptation_steps,
                )
                distance_stage = get_distance_stage(
                    train_env.reset_max_path_length,
                    args.hard_path_length_threshold,
                )
                if full_random_adapt_step is not None:
                    epsilon = min(
                        1.0,
                        epsilon + linear_schedule(
                            step=full_random_adapt_step,
                            start=args.full_random_epsilon_boost,
                            end=0.0,
                            duration=args.full_random_adaptation_steps,
                        ),
                    )
                    temperature = temperature + linear_schedule(
                        step=full_random_adapt_step,
                        start=args.full_random_temperature_boost,
                        end=0.0,
                        duration=args.full_random_adaptation_steps,
                    )
                if distance_stage == 1:
                    epsilon = min(1.0, epsilon + args.hard_path_epsilon_boost)
                    temperature = temperature + args.hard_path_temperature_boost
                elif distance_stage == 2:
                    epsilon = min(1.0, epsilon + args.hard_path_stage2_epsilon_boost)
                    temperature = temperature + args.hard_path_stage2_temperature_boost
                elif distance_stage >= 3:
                    epsilon = min(1.0, epsilon + args.hard_path_stage3_epsilon_boost)
                    temperature = temperature + args.hard_path_stage3_temperature_boost
                episode = collect_episode_actor_policy(
                    env=train_env,
                    encoder=encoder,
                    rssm=rssm,
                    actor=actor,
                    device=device,
                    max_steps=args.max_episode_steps,
                    epsilon=epsilon,
                    temperature=temperature,
                    deterministic=False,
                    exploration_action_probs=random_action_probs,
                )
                last_collect_mode = f"actor_eps={epsilon:.3f}_temp={temperature:.3f}"
                if full_random_adapt_step is not None:
                    last_collect_mode += "_fr_adapt"
                if distance_stage > 0:
                    last_collect_mode += f"_hard_s{distance_stage}"

            add_episode_to_buffer(replay_buffer, episode)

        # World model updates.
        for _ in range(args.model_updates_per_step):
            batch = replay_buffer.sample_batch(
                batch_size=args.batch_size,
                seq_len=args.seq_len,
            )
            obs_tensor, actions_tensor, reward_targets, done_targets, event_targets, valid_mask = batch_to_tensors(
                batch,
                device=device,
            )

            wm_loss = train_world_model_step(
                encoder=encoder,
                rssm=rssm,
                decoder=decoder,
                reward_model=reward_model,
                continue_model=continue_model,
                event_model=event_model,
                optimizer=world_model_optimizer,
                obs_tensor=obs_tensor,
                actions_tensor=actions_tensor,
                reward_targets=reward_targets,
                done_targets=done_targets,
                event_targets=event_targets,
                valid_mask=valid_mask,
                recon_scale=1.0,
                reward_scale=1.0,
                continue_scale=1.0,
                event_scale=1.0,
                kl_scale=0.1,
                free_nats=1.0,
                dyn_scale=0.5,
                rep_scale=0.1,
                continue_terminal_weight=8.0,
                prior_loss_scale=0.5,
                success_reward_weight=args.success_reward_weight,
                terminal_reward_weight=args.terminal_reward_weight,
                event_positive_weights=torch.tensor([6.0, 10.0], device=device),
                grad_clip=100.0,
            )

        # Actor-Critic updates from latent imagination.
        ac_loss = None
        bc_loss = None
        if step >= args.actor_start_step:
            for _ in range(args.actor_updates_per_step):
                # Re-use the last sampled batch posterior states.
                init_indices = sample_imagination_init_indices(
                    valid_mask=valid_mask,
                    done_targets=done_targets,
                    collision_targets=event_targets[..., 1:2],
                )
                init_state = select_state_by_index(
                    wm_loss.posteriors,
                    init_indices,
                )

                # Early AC uses shorter horizon and stronger entropy to reduce
                # model-bias exploitation.
                if step < args.actor_start_step + 20000:
                    horizon = min(args.imagination_horizon, 10)
                    entropy_scale = 3e-2
                else:
                    horizon = args.imagination_horizon
                    entropy_scale = 1e-2

                full_random_adapt_step = get_full_random_adaptation_progress(
                    step=step,
                    start_step=args.reset_curriculum_start_step,
                    duration=args.reset_curriculum_duration,
                    hold_end_steps=args.curriculum_hold_end_steps,
                    adaptation_steps=args.full_random_adaptation_steps,
                )
                distance_stage = get_distance_stage(
                    train_env.reset_max_path_length,
                    args.hard_path_length_threshold,
                )
                if full_random_adapt_step is not None:
                    horizon = min(horizon, args.full_random_adaptation_horizon)
                    entropy_scale = max(
                        entropy_scale,
                        args.full_random_adaptation_entropy_scale,
                    )

                current_real_critic_scale = args.real_critic_scale
                if distance_stage == 1:
                    horizon = min(horizon, args.hard_path_horizon)
                    entropy_scale = max(entropy_scale, args.hard_path_entropy_scale)
                    current_real_critic_scale = max(
                        current_real_critic_scale,
                        args.hard_path_real_critic_scale,
                    )
                elif distance_stage == 2:
                    horizon = min(horizon, args.hard_path_stage2_horizon)
                    entropy_scale = max(entropy_scale, args.hard_path_stage2_entropy_scale)
                    current_real_critic_scale = max(
                        current_real_critic_scale,
                        args.hard_path_stage2_real_critic_scale,
                    )
                elif distance_stage >= 3:
                    horizon = min(horizon, args.hard_path_stage3_horizon)
                    entropy_scale = max(entropy_scale, args.hard_path_stage3_entropy_scale)
                    current_real_critic_scale = max(
                        current_real_critic_scale,
                        args.hard_path_stage3_real_critic_scale,
                    )

                if step >= collapse_ac_pause_until and step % args.actor_ac_interval == 0:
                    ac_loss = train_actor_critic_step(
                    rssm=rssm,
                    actor=actor,
                    reward_model=reward_model,
                    continue_model=continue_model,
                    event_model=event_model,
                    critic=critic,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    init_state=init_state,
                    horizon=horizon,
                    gamma=0.99,
                    lambda_=0.95,
                    entropy_scale=entropy_scale,
                    critic_scale=1.0,
                    continue_return_scale=1.0,
                    reward_clip=6.0,
                    success_bonus=4.0,
                    collision_penalty=max(10.0, args.collision_penalty),
                    collision_continue_scale=args.ac_collision_continue_scale,
                    non_forward_penalty_scale=args.ac_non_forward_penalty,
                    forward_action_index=GridNavEnv.ACTION_FORWARD,
                    real_feat=wm_loss.posterior_feat,
                    real_rewards=reward_targets,
                    real_dones=done_targets,
                    real_valid_mask=valid_mask,
                    real_critic_scale=current_real_critic_scale,
                    grad_clip=100.0,
                    )

                if args.actor_bc_scale > 0.0 and replay_buffer.num_success_episodes > 0:
                    for _ in range(args.actor_bc_updates_per_step):
                        success_batch = replay_buffer.sample_success_batch(
                            batch_size=args.batch_size,
                            seq_len=args.seq_len,
                        )
                        (
                            success_obs_tensor,
                            success_actions_tensor,
                            _,
                            _,
                            _,
                            success_valid_mask,
                        ) = batch_to_tensors(success_batch, device=device)
                        bc_loss = train_actor_bc_step(
                            encoder=encoder,
                            rssm=rssm,
                            actor=actor,
                            optimizer=actor_optimizer,
                            obs_tensor=success_obs_tensor,
                            actions_tensor=success_actions_tensor,
                            valid_mask=success_valid_mask,
                            loss_scale=args.actor_bc_scale * collapse_bc_scale_multiplier,
                            grad_clip=100.0,
                        )

        # Logging.
        if step % args.log_interval == 0:
            with torch.no_grad():
                continue_prob = torch.sigmoid(wm_loss.pred_continue_logits)
                valid_continue_prob = continue_prob[valid_mask > 0.5]
                valid_done = done_targets[valid_mask > 0.5]
                avg_continue_prob = valid_continue_prob.mean().item()
                terminal_mask = valid_done > 0.5
                terminal_continue_prob = (
                    valid_continue_prob[terminal_mask].mean().item()
                    if terminal_mask.any()
                    else -1.0
                )

            if step < args.actor_start_step:
                phase = "world_model_only"
            else:
                phase = "world_model+imagined_actor_critic"

            log_text = (
                f"[step {step:06d}] phase={phase}, mode={last_collect_mode}, reset={reset_mode}, unlocked_max={unlocked_max_path_length:02d}, "
                f"buffer_ep={len(replay_buffer):04d}, regular_ep={replay_buffer.num_regular_episodes:04d}, "
                f"success_ep={replay_buffer.num_success_episodes:04d}, success_path_max={replay_buffer.max_success_path_length:02d}, buffer_steps={replay_buffer.num_steps:06d}, "
                f"wm={wm_loss.total_loss.item():.4f}, recon={wm_loss.recon_loss.item():.4f}, "
                f"reward={wm_loss.reward_loss.item():.4f}, cont={wm_loss.continue_loss.item():.4f}, "
                f"event={wm_loss.event_loss.item():.4f}, kl={wm_loss.kl_loss.item():.4f}, cont_acc={wm_loss.continue_acc.item():.3f}, "
                f"event_acc={wm_loss.event_acc.item():.3f}, "
                f"cont_p={avg_continue_prob:.3f}, term_cont_p={terminal_continue_prob:.3f}"
            )

            if ac_loss is not None:
                imag_action_hist = [float(x) for x in ac_loss.action_histogram.detach().cpu().tolist()]
                log_text += (
                    f", actor={ac_loss.actor_loss.item():.4f}, "
                    f"critic={ac_loss.critic_loss.item():.4f}, "
                    f"imag_ret={ac_loss.mean_return.item():.4f}, "
                    f"imag_rew={ac_loss.mean_reward.item():.4f}, "
                    f"imag_v={ac_loss.mean_value.item():.4f}, "
                    f"imag_cont={ac_loss.mean_continue.item():.3f}, "
                    f"imag_act={imag_action_hist}, "
                    f"critic_imag={ac_loss.imagined_critic_loss.item():.4f}, "
                    f"critic_real={ac_loss.real_critic_loss.item():.4f}"
                )
            if bc_loss is not None:
                bc_action_hist = [float(x) for x in bc_loss.action_histogram.detach().cpu().tolist()]
                log_text += (
                    f", bc={bc_loss.bc_loss.item():.4f}, "
                    f"bc_acc={bc_loss.bc_acc.item():.3f}, "
                    f"bc_act={bc_action_hist}"
                )

            print(log_text)

        # Evaluation.
        if step % args.eval_interval == 0:
            if step < args.actor_start_step:
                print(
                    f"\n[Actor Eval step {step:06d}] skipped: "
                    f"actor has not started training yet "
                    f"(actor_start_step={args.actor_start_step}).\n"
                )
            else:
                eval_reset_mode = configure_reset_curriculum(
                    env=eval_env,
                    step=step,
                    start_step=args.reset_curriculum_start_step,
                    duration=args.reset_curriculum_duration,
                    min_path_length=args.curriculum_min_path_length,
                    start_max_path_length=args.curriculum_start_max_path_length,
                    end_max_path_length=args.curriculum_end_max_path_length,
                    hold_end_steps=args.curriculum_hold_end_steps,
                    unlocked_max_path_length=unlocked_max_path_length,
                )
                metrics_det = evaluate_actor_agent(
                    env=eval_env,
                    encoder=encoder,
                    rssm=rssm,
                    actor=actor,
                    device=device,
                    num_episodes=args.eval_episodes,
                    max_steps=args.max_episode_steps,
                    deterministic=True,
                    temperature=1.0,
                )
                metrics_sto = evaluate_actor_agent(
                    env=eval_env,
                    encoder=encoder,
                    rssm=rssm,
                    actor=actor,
                    device=device,
                    num_episodes=args.eval_episodes,
                    max_steps=args.max_episode_steps,
                    deterministic=False,
                    temperature=1.0,
                )

                print(
                    f"\n[Actor Eval step {step:06d}] reset={eval_reset_mode}\n"
                    f"  det: avg_reward={metrics_det['avg_reward']:.3f}, success={metrics_det['success_rate']:.3f}, "
                    f"collision={metrics_det['collision_rate']:.3f}, avg_length={metrics_det['avg_length']:.1f}, "
                    f"action_counts={metrics_det['action_counts']}, avg_probs={[round(x, 3) for x in metrics_det['avg_probs']]}\n"
                    f"  sto: avg_reward={metrics_sto['avg_reward']:.3f}, success={metrics_sto['success_rate']:.3f}, "
                    f"collision={metrics_sto['collision_rate']:.3f}, avg_length={metrics_sto['avg_length']:.1f}, "
                    f"action_counts={metrics_sto['action_counts']}, avg_probs={[round(x, 3) for x in metrics_sto['avg_probs']]}\n"
                )

                eval_max_path_length = eval_env.reset_max_path_length
                if (
                    eval_max_path_length is not None
                    and eval_max_path_length == unlocked_max_path_length
                    and unlocked_max_path_length < args.curriculum_end_max_path_length
                    and metrics_det["success_rate"] >= args.curriculum_advance_success_threshold
                ):
                    curriculum_success_streak += 1
                else:
                    curriculum_success_streak = 0

                if curriculum_success_streak >= args.curriculum_advance_min_evals:
                    unlocked_max_path_length = min(
                        unlocked_max_path_length + 1,
                        args.curriculum_end_max_path_length,
                    )
                    curriculum_success_streak = 0
                    print(
                        f"[Curriculum Gate] unlocked_max_path_length -> {unlocked_max_path_length}"
                    )

                if metrics_det["success_rate"] > best_success:
                    best_success = metrics_det["success_rate"]
                    best_actor_state = copy.deepcopy(actor.state_dict())
                    best_critic_state = copy.deepcopy(critic.state_dict())
                    collapse_bc_scale_multiplier = 1.0
                    best_path = (
                        ROOT_DIR
                        / "outputs"
                        / "checkpoints_standard_wmrl"
                        / "world_model_actor_critic_best.pt"
                    )
                    save_checkpoint(
                        save_path=best_path,
                        encoder=encoder,
                        rssm=rssm,
                        decoder=decoder,
                        reward_model=reward_model,
                        continue_model=continue_model,
                        event_model=event_model,
                        actor=actor,
                        critic=critic,
                        step=step,
                        best_success=best_success,
                    )
                    print(f"[Best Checkpoint] saved to: {best_path}")

                should_restore = (
                    best_actor_state is not None
                    and best_success >= args.collapse_restore_min_best_success
                    and metrics_det["success_rate"] <= best_success - args.collapse_restore_threshold
                    and metrics_det["collision_rate"] >= args.collapse_restore_collision_threshold
                )

                if should_restore:
                    actor.load_state_dict(best_actor_state)
                    critic.load_state_dict(best_critic_state)
                    collapse_ac_pause_until = step + args.collapse_ac_pause_steps
                    collapse_bc_scale_multiplier = max(collapse_bc_scale_multiplier, args.collapse_bc_boost)
                    for group in actor_optimizer.param_groups:
                        group["lr"] = max(group["lr"] * args.collapse_actor_lr_decay, 1e-6)
                    print(
                        "[Collapse Restore] restored best actor/critic, "
                        f"pause_ac_until={collapse_ac_pause_until}, "
                        f"actor_lr={actor_optimizer.param_groups[0]['lr']:.2e}, "
                        f"bc_scale_multiplier={collapse_bc_scale_multiplier:.2f}"
                    )
                elif step >= collapse_ac_pause_until and collapse_bc_scale_multiplier > 1.0:
                    collapse_bc_scale_multiplier = 1.0

        # Periodic checkpoint.
        if step % args.save_interval == 0:
            save_path = (
                ROOT_DIR
                / "outputs"
                / "checkpoints_standard_wmrl"
                / f"world_model_actor_critic_step_{step}.pt"
            )
            save_checkpoint(
                save_path=save_path,
                encoder=encoder,
                rssm=rssm,
                decoder=decoder,
                reward_model=reward_model,
                continue_model=continue_model,
                event_model=event_model,
                actor=actor,
                critic=critic,
                step=step,
                best_success=best_success,
            )
            print(f"[Checkpoint] saved to: {save_path}")

    final_path = (
        ROOT_DIR
        / "outputs"
        / "checkpoints_standard_wmrl"
        / "world_model_actor_critic_final.pt"
    )
    save_checkpoint(
        save_path=final_path,
        encoder=encoder,
        rssm=rssm,
        decoder=decoder,
        reward_model=reward_model,
        continue_model=continue_model,
        event_model=event_model,
        actor=actor,
        critic=critic,
        step=args.total_train_steps,
        best_success=best_success,
    )

    print("\nTraining finished.")
    print(f"Final checkpoint: {final_path}")
    print(f"Best success during training: {best_success:.3f}")


if __name__ == "__main__":
    main()
