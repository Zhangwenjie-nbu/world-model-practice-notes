# scripts/test_actor_eval.py

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from envs.grid_nav_env import GridNavEnv, collect_episode_heuristic
from models.actor import DiscreteActor
from models.encoder import VisualEncoder
from models.rssm import RSSM
from tools.seed import set_global_seed


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(obs).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x.to(device)


def load_models(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    encoder = VisualEncoder(image_size=64, in_channels=3, embedding_dim=256).to(device)
    rssm = RSSM(
        embedding_dim=256,
        num_actions=4,
        action_embed_dim=32,
        deter_dim=256,
        stoch_dim=32,
        hidden_dim=256,
        min_std=0.1,
    ).to(device)
    actor = DiscreteActor(feature_dim=288, hidden_dim=256, num_actions=4).to(device)

    encoder.load_state_dict(ckpt['encoder'])
    rssm.load_state_dict(ckpt['rssm'])
    if 'actor' not in ckpt:
        raise KeyError(f'checkpoint does not contain actor weights: {checkpoint_path}')
    actor.load_state_dict(ckpt['actor'])

    encoder.eval()
    rssm.eval()
    actor.eval()

    return encoder, rssm, actor


@torch.no_grad()
def evaluate_actor_agent(
    env: GridNavEnv,
    encoder: VisualEncoder,
    rssm: RSSM,
    actor: DiscreteActor,
    device: torch.device,
    num_episodes: int = 10,
    max_steps: int = 100,
    deterministic: bool = True,
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

    for _ in range(num_episodes):
        obs = env.reset()
        state = rssm.initial_state(batch_size=1, device=device)

        obs_tensor = obs_to_tensor(obs, device)
        embedding = encoder(obs_tensor)
        dummy_action = torch.tensor([GridNavEnv.ACTION_STAY], dtype=torch.long, device=device)
        state, _ = rssm.obs_step(prev_state=state, action=dummy_action, embedding=embedding)

        episode_reward = 0.0
        success = False
        collision = False
        stuck = False
        length = 0

        for _ in range(max_steps):
            feat = torch.cat([state.deter, state.mean], dim=-1)
            dist = actor(feat, temperature=temperature)
            probs = dist.probs.squeeze(0).detach().cpu().numpy()
            prob_sum += probs
            prob_count += 1

            if deterministic:
                action = int(torch.argmax(dist.probs, dim=-1).item())
            else:
                action = int(dist.sample().item())
            action_counts[action] += 1

            next_obs, reward, done, info = env.step(action)
            episode_reward += float(reward)
            length += 1

            success = bool(info.get('success', False))
            collision = bool(info.get('collision', False))
            stuck = bool(info.get('stuck', False))

            if done:
                break

            next_obs_tensor = obs_to_tensor(next_obs, device)
            next_embedding = encoder(next_obs_tensor)
            action_tensor = torch.tensor([action], dtype=torch.long, device=device)
            state, _ = rssm.obs_step(prev_state=state, action=action_tensor, embedding=next_embedding)
            obs = next_obs

        total_rewards.append(episode_reward)
        successes.append(float(success))
        collisions.append(float(collision))
        stucks.append(float(stuck))
        lengths.append(float(length))

    return {
        'avg_reward': float(np.mean(total_rewards)),
        'success_rate': float(np.mean(successes)),
        'collision_rate': float(np.mean(collisions)),
        'stuck_rate': float(np.mean(stucks)),
        'avg_length': float(np.mean(lengths)),
        'action_counts': action_counts.tolist(),
        'avg_action_probs': (prob_sum / max(prob_count, 1)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT_DIR / 'outputs' / 'checkpoints' / 'world_model_critic_actor_final.pt')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample', action='store_true', help='sample actions instead of argmax')
    args = parser.parse_args()

    set_global_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, rssm, actor = load_models(args.checkpoint, device)

    actor_env = GridNavEnv(map_width=10, map_height=10, image_size=64, max_steps=args.max_steps, random_reset=True, seed=999)
    heuristic_env = GridNavEnv(map_width=10, map_height=10, image_size=64, max_steps=args.max_steps, random_reset=True, seed=2024)

    heuristic_results = []
    for _ in range(args.episodes):
        ep = collect_episode_heuristic(env=heuristic_env, max_steps=args.max_steps)
        heuristic_results.append(ep)

    heuristic_avg = {
        'avg_reward': float(np.mean([ep['total_reward'] for ep in heuristic_results])),
        'success_rate': float(np.mean([float(ep['success']) for ep in heuristic_results])),
        'collision_rate': float(np.mean([float(ep['collision']) for ep in heuristic_results])),
        'stuck_rate': float(np.mean([float(ep['stuck']) for ep in heuristic_results])),
        'avg_length': float(np.mean([float(ep['length']) for ep in heuristic_results])),
    }

    actor_metrics = evaluate_actor_agent(
        env=actor_env,
        encoder=encoder,
        rssm=rssm,
        actor=actor,
        device=device,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        deterministic=not args.sample,
        temperature=args.temperature,
    )

    print(f'checkpoint={args.checkpoint}')
    print(f"policy_mode={'sample' if args.sample else 'argmax'}, temperature={args.temperature:.3f}")
    print(
        f"[heuristic] avg_reward={heuristic_avg['avg_reward']:.3f}, success={heuristic_avg['success_rate']:.3f}, "
        f"collision={heuristic_avg['collision_rate']:.3f}, stuck={heuristic_avg['stuck_rate']:.3f}, "
        f"avg_length={heuristic_avg['avg_length']:.1f}"
    )
    print(
        f"[actor] avg_reward={actor_metrics['avg_reward']:.3f}, success={actor_metrics['success_rate']:.3f}, "
        f"collision={actor_metrics['collision_rate']:.3f}, stuck={actor_metrics['stuck_rate']:.3f}, "
        f"avg_length={actor_metrics['avg_length']:.1f}, action_counts={actor_metrics['action_counts']}, "
        f"avg_probs={[round(x, 3) for x in actor_metrics['avg_action_probs']]}"
    )


if __name__ == '__main__':
    main()
