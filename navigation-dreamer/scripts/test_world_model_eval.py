# scripts/test_world_model_eval.py

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from envs.grid_nav_env import GridNavEnv
from memory.replay_buffer import EpisodeReplayBuffer
from models.continue_model import ContinueModel
from models.decoder import VisualDecoder
from models.encoder import VisualEncoder
from models.event_model import EventModel
from models.reward_model import RewardModel
from models.rssm import RSSM
from scripts.train_agent import (
    batch_to_tensors,
    collect_episode_collision_probe,
    collect_episode_heuristic,
    collect_episode_noisy_heuristic,
    collect_episode_turn_loop_probe,
)
from tools.seed import set_global_seed
from trainers.joint_trainer import compute_world_model_loss


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

    encoder.load_state_dict(ckpt['encoder'])
    rssm.load_state_dict(ckpt['rssm'])
    decoder.load_state_dict(ckpt['decoder'])
    reward_model.load_state_dict(ckpt['reward_model'])
    continue_model.load_state_dict(ckpt['continue_model'])
    event_model.load_state_dict(ckpt['event_model'])

    encoder.eval()
    rssm.eval()
    decoder.eval()
    reward_model.eval()
    continue_model.eval()
    event_model.eval()

    return encoder, rssm, decoder, reward_model, continue_model, event_model


def collect_to_buffer(env, collector, num_episodes: int, max_steps: int):
    buffer = EpisodeReplayBuffer(capacity_episodes=num_episodes + 8, padding_action=GridNavEnv.ACTION_STAY)
    for _ in range(num_episodes):
        episode = collector(env=env, max_steps=max_steps)
        buffer.add_episode(
            obs=episode['obs'],
            actions=episode['actions'],
            rewards=episode['rewards'],
            dones=episode['dones'],
            successes=episode.get('successes', None),
            collisions=episode.get('collisions', None),
            stucks=episode.get('stucks', None),
        )
    return buffer


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


@torch.no_grad()
def evaluate_buffer(
    buffer: EpisodeReplayBuffer,
    encoder,
    rssm,
    decoder,
    reward_model,
    continue_model,
    event_model,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    num_batches: int,
    continue_terminal_weight: float,
    event_positive_weights: torch.Tensor,
):
    totals = {
        'wm': 0.0,
        'recon': 0.0,
        'reward': 0.0,
        'cont': 0.0,
        'event': 0.0,
        'kl': 0.0,
        'cont_acc': 0.0,
        'event_acc': 0.0,
        'term_cont': 0.0,
        'term_count': 0,
        'valid_count': 0,
    }
    event_tp = torch.zeros(3, dtype=torch.float64)
    event_fp = torch.zeros(3, dtype=torch.float64)
    event_fn = torch.zeros(3, dtype=torch.float64)

    for _ in range(num_batches):
        batch = buffer.sample_batch(batch_size=batch_size, seq_len=seq_len)
        obs_tensor, actions_tensor, reward_targets, done_targets, event_targets, valid_mask = batch_to_tensors(batch, device=device)

        loss = compute_world_model_loss(
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
            recon_scale=1.0,
            reward_scale=1.0,
            continue_scale=1.0,
            event_scale=1.0,
            kl_scale=0.1,
            free_nats=1.0,
            dyn_scale=0.5,
            rep_scale=0.1,
            continue_terminal_weight=continue_terminal_weight,
            event_positive_weights=event_positive_weights,
        )

        totals['wm'] += float(loss.total_loss.item())
        totals['recon'] += float(loss.recon_loss.item())
        totals['reward'] += float(loss.reward_loss.item())
        totals['cont'] += float(loss.continue_loss.item())
        totals['event'] += float(loss.event_loss.item())
        totals['kl'] += float(loss.kl_loss.item())
        totals['cont_acc'] += float(loss.continue_acc.item())
        totals['event_acc'] += float(loss.event_acc.item())

        cont_probs = torch.sigmoid(loss.pred_continue_logits)
        valid = valid_mask > 0.5
        done = done_targets > 0.5
        term_mask = valid & done
        if term_mask.any():
            totals['term_cont'] += float(cont_probs[term_mask].mean().item())
            totals['term_count'] += 1
        totals['valid_count'] += 1

        event_probs = torch.sigmoid(loss.pred_event_logits)
        event_pred = (event_probs >= 0.5).float()
        event_tgt = event_targets.float()
        mask = valid_mask.expand_as(event_tgt) > 0.5
        for i in range(3):
            p = event_pred[..., i]
            t = event_tgt[..., i]
            m = mask[..., i]
            event_tp[i] += ((p == 1) & (t == 1) & m).sum().item()
            event_fp[i] += ((p == 1) & (t == 0) & m).sum().item()
            event_fn[i] += ((p == 0) & (t == 1) & m).sum().item()

    n = float(num_batches)
    event_precision = [safe_div(event_tp[i], event_tp[i] + event_fp[i]) for i in range(3)]
    event_recall = [safe_div(event_tp[i], event_tp[i] + event_fn[i]) for i in range(3)]
    event_f1 = [safe_div(2 * event_precision[i] * event_recall[i], event_precision[i] + event_recall[i]) for i in range(3)]

    return {
        'wm': totals['wm'] / n,
        'recon': totals['recon'] / n,
        'reward': totals['reward'] / n,
        'cont': totals['cont'] / n,
        'event': totals['event'] / n,
        'kl': totals['kl'] / n,
        'cont_acc': totals['cont_acc'] / n,
        'event_acc': totals['event_acc'] / n,
        'term_cont': totals['term_cont'] / max(totals['term_count'], 1),
        'event_precision': event_precision,
        'event_recall': event_recall,
        'event_f1': event_f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT_DIR / 'outputs' / 'checkpoints' / 'world_model_final.pt')
    parser.add_argument('--episodes', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--num-batches', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=100)
    args = parser.parse_args()

    set_global_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, rssm, decoder, reward_model, continue_model, event_model = load_world_model(args.checkpoint, device)

    eval_env = GridNavEnv(map_width=10, map_height=10, image_size=64, max_steps=args.max_steps, random_reset=True, seed=999)
    event_positive_weights = torch.tensor([2.0, 4.0, 4.0], device=device)
    continue_terminal_weight = 8.0

    splits = [
        ('heuristic', collect_episode_heuristic),
        ('noisy', collect_episode_noisy_heuristic),
        ('collision_probe', collect_episode_collision_probe),
        ('turn_loop_probe', collect_episode_turn_loop_probe),
    ]

    print(f'checkpoint={args.checkpoint}')
    for name, collector in splits:
        buffer = collect_to_buffer(eval_env, collector, args.episodes, args.max_steps)
        metrics = evaluate_buffer(
            buffer=buffer,
            encoder=encoder,
            rssm=rssm,
            decoder=decoder,
            reward_model=reward_model,
            continue_model=continue_model,
            event_model=event_model,
            device=device,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_batches=args.num_batches,
            continue_terminal_weight=continue_terminal_weight,
            event_positive_weights=event_positive_weights,
        )
        print(
            f'[{name}] wm={metrics["wm"]:.4f}, recon={metrics["recon"]:.4f}, reward={metrics["reward"]:.4f}, '
            f'cont={metrics["cont"]:.4f}, event={metrics["event"]:.4f}, kl={metrics["kl"]:.4f}, '
            f'cont_acc={metrics["cont_acc"]:.3f}, event_acc={metrics["event_acc"]:.3f}, '
            f'term_cont={metrics["term_cont"]:.3f}'
        )
        print(
            f'  event_precision={[round(x, 3) for x in metrics["event_precision"]]}, '
            f'event_recall={[round(x, 3) for x in metrics["event_recall"]]}, '
            f'event_f1={[round(x, 3) for x in metrics["event_f1"]]}'
        )


if __name__ == '__main__':
    main()
