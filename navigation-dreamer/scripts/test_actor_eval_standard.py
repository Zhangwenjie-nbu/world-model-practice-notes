# scripts/test_actor_eval_standard.py

"""
Evaluate a standard world-model RL actor checkpoint.

This evaluator matches checkpoints saved by:
    scripts/train_agent_standard_wmrl.py

It does not require EventModel, expert buffer, BC, or DAgger.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from envs.grid_nav_env import GridNavEnv
from models.actor import DiscreteActor
from models.continue_model import ContinueModel
from models.critic import Critic
from models.decoder import VisualDecoder
from models.encoder import VisualEncoder
from models.reward_model import RewardModel
from models.rssm import RSSM
from tools.seed import set_global_seed
from trainers.joint_trainer_standard import get_actor_feat_from_state


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(obs).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x.to(device)


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

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
    actor = DiscreteActor(feature_dim=288, hidden_dim=256, num_actions=3).to(device)
    critic = Critic(feature_dim=288, hidden_dim=256).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    rssm.load_state_dict(ckpt["rssm"])
    decoder.load_state_dict(ckpt["decoder"])
    reward_model.load_state_dict(ckpt["reward_model"])
    continue_model.load_state_dict(ckpt["continue_model"])
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])

    encoder.eval()
    rssm.eval()
    decoder.eval()
    reward_model.eval()
    continue_model.eval()
    actor.eval()
    critic.eval()

    return encoder, rssm, decoder, reward_model, continue_model, actor, critic


@torch.no_grad()
def evaluate_actor(
    env: GridNavEnv,
    encoder: VisualEncoder,
    rssm: RSSM,
    actor: DiscreteActor,
    device: torch.device,
    episodes: int,
    max_steps: int,
    policy_mode: str = "argmax",
    temperature: float = 1.0,
):
    total_rewards = []
    successes = []
    collisions = []
    stucks = []
    lengths = []
    action_counts = np.zeros(env.num_actions, dtype=np.int64)
    prob_sum = np.zeros(env.num_actions, dtype=np.float64)
    prob_count = 0

    for _ in range(episodes):
        obs = env.reset()
        state = rssm.initial_state(batch_size=1, device=device)
        prev_action = GridNavEnv.ACTION_FORWARD

        total_reward = 0.0
        success = False
        collision = False
        stuck = False
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

            if policy_mode == "argmax":
                action = int(np.argmax(probs))
            elif policy_mode == "sample":
                action = int(dist.sample().item())
            else:
                raise ValueError(f"Unsupported policy_mode: {policy_mode}")

            action_counts[action] += 1

            next_obs, reward, done, info = env.step(action)

            total_reward += float(reward)
            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            stuck = bool(info.get("stuck", False))
            length += 1

            prev_action = action
            obs = next_obs

            if done:
                break

        total_rewards.append(total_reward)
        successes.append(float(success))
        collisions.append(float(collision))
        stucks.append(float(stuck))
        lengths.append(float(length))

    avg_probs = prob_sum / max(prob_count, 1)

    return {
        "avg_reward": float(np.mean(total_rewards)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "stuck_rate": float(np.mean(stucks)),
        "avg_length": float(np.mean(lengths)),
        "action_counts": action_counts.tolist(),
        "avg_probs": [float(x) for x in avg_probs],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT_DIR
        / "outputs"
        / "checkpoints_standard_wmrl"
        / "world_model_actor_critic_best.pt",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--policy-mode", type=str, default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--terminate-on-collision",
        action="store_true",
        default=False,
        help=(
            "If set, collision immediately terminates the episode. "
            "By default this evaluator matches the standard WM-RL training "
            "setting where collision is penalized but does not end the episode."
        ),
    )
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    set_global_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, rssm, _, _, _, actor, _ = load_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    env = GridNavEnv(
        map_width=10,
        map_height=10,
        image_size=64,
        max_steps=args.max_steps,
        random_reset=True,
        terminate_on_collision=args.terminate_on_collision,
        seed=args.seed,
    )

    metrics = evaluate_actor(
        env=env,
        encoder=encoder,
        rssm=rssm,
        actor=actor,
        device=device,
        episodes=args.episodes,
        max_steps=args.max_steps,
        policy_mode=args.policy_mode,
        temperature=args.temperature,
    )

    print(f"checkpoint={args.checkpoint}")
    print(
        f"policy_mode={args.policy_mode}, temperature={args.temperature:.3f}, "
        f"terminate_on_collision={args.terminate_on_collision}"
    )
    print(
        f"[actor] avg_reward={metrics['avg_reward']:.3f}, "
        f"success={metrics['success_rate']:.3f}, "
        f"collision={metrics['collision_rate']:.3f}, "
        f"stuck={metrics['stuck_rate']:.3f}, "
        f"avg_length={metrics['avg_length']:.1f}, "
        f"action_counts={metrics['action_counts']}, "
        f"avg_probs={[round(x, 3) for x in metrics['avg_probs']]}"
    )


if __name__ == "__main__":
    main()
