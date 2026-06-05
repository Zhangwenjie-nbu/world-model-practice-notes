# scripts/test_planner_eval.py

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from envs.grid_nav_env import GridNavEnv, collect_episode_heuristic
from models.continue_model import ContinueModel
from models.critic import Critic
from models.decoder import VisualDecoder
from models.encoder import VisualEncoder
from models.event_model import EventModel
from models.planner import select_action_by_random_shooting_continue
from models.reward_model import RewardModel
from models.rssm import RSSM
from tools.seed import set_global_seed


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(obs).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x.to(device)


def load_world_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

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
    decoder = VisualDecoder(
        feature_dim=288,
        image_size=64,
        out_channels=3,
        hidden_channels=256,
    ).to(device)
    reward_model = RewardModel(feature_dim=288, hidden_dim=256).to(device)
    continue_model = ContinueModel(feature_dim=288, hidden_dim=256).to(device)
    event_model = EventModel(feature_dim=288, hidden_dim=256, num_events=3).to(device)
    critic = Critic(feature_dim=288, hidden_dim=256).to(device)

    encoder.load_state_dict(ckpt['encoder'])
    rssm.load_state_dict(ckpt['rssm'])
    decoder.load_state_dict(ckpt['decoder'])
    reward_model.load_state_dict(ckpt['reward_model'])
    continue_model.load_state_dict(ckpt['continue_model'])
    event_model.load_state_dict(ckpt['event_model'])
    if 'critic' in ckpt:
        critic.load_state_dict(ckpt['critic'])

    encoder.eval()
    rssm.eval()
    decoder.eval()
    reward_model.eval()
    continue_model.eval()
    event_model.eval()
    critic.eval()

    return encoder, rssm, decoder, reward_model, continue_model, event_model, critic


@torch.no_grad()
def evaluate_planner_agent(
    env: GridNavEnv,
    encoder: VisualEncoder,
    rssm: RSSM,
    reward_model: RewardModel,
    continue_model: ContinueModel,
    event_model: EventModel,
    critic: Critic,
    device: torch.device,
    num_episodes: int = 10,
    max_steps: int = 100,
    horizon: int = 12,
    num_candidates: int = 1024,
):
    total_rewards = []
    successes = []
    collisions = []
    stucks = []
    lengths = []
    action_counts = np.zeros(env.num_actions, dtype=np.int64)

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
            plan = select_action_by_random_shooting_continue(
                rssm=rssm,
                reward_model=reward_model,
                continue_model=continue_model,
                critic=critic,
                init_state=state,
                num_actions=env.num_actions,
                horizon=horizon,
                num_candidates=num_candidates,
                gamma=0.99,
                action_repeat_penalty=0.03,
                done_risk_penalty=3.0,
                terminal_value_scale=0.35,
                event_model=event_model,
                success_bonus=8.0,
                collision_penalty=6.0,
                stuck_penalty=5.0,
            )

            action = int(plan.best_action.item())
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT_DIR / 'outputs' / 'checkpoints' / 'world_model_critic_actor_final.pt')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--num-candidates', type=int, default=1024)
    args = parser.parse_args()

    set_global_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, rssm, _, reward_model, continue_model, event_model, critic = load_world_model(args.checkpoint, device)

    planner_env = GridNavEnv(map_width=10, map_height=10, image_size=64, max_steps=args.max_steps, random_reset=True, seed=999)
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

    planner_metrics = evaluate_planner_agent(
        env=planner_env,
        encoder=encoder,
        rssm=rssm,
        reward_model=reward_model,
        continue_model=continue_model,
        event_model=event_model,
        critic=critic,
        device=device,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
    )

    print(f'checkpoint={args.checkpoint}')
    print(
        f"[heuristic] avg_reward={heuristic_avg['avg_reward']:.3f}, success={heuristic_avg['success_rate']:.3f}, "
        f"collision={heuristic_avg['collision_rate']:.3f}, stuck={heuristic_avg['stuck_rate']:.3f}, "
        f"avg_length={heuristic_avg['avg_length']:.1f}"
    )
    print(
        f"[planner] avg_reward={planner_metrics['avg_reward']:.3f}, success={planner_metrics['success_rate']:.3f}, "
        f"collision={planner_metrics['collision_rate']:.3f}, stuck={planner_metrics['stuck_rate']:.3f}, "
        f"avg_length={planner_metrics['avg_length']:.1f}, action_counts={planner_metrics['action_counts']}"
    )


if __name__ == '__main__':
    main()
