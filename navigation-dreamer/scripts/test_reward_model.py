# scripts/test_reward_model.py

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 让脚本可以从项目根目录导入模块
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from envs.grid_nav_env import GridNavEnv
from models.encoder import VisualEncoder
from models.rssm import RSSM
from models.reward_model import RewardModel
from models.common import count_parameters


def obs_sequence_to_tensor(obs_sequence: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    将环境图像序列转换为 PyTorch Tensor。

    输入：
        obs_sequence:
            形状为 (T, H, W, C)，dtype=uint8，range=[0,255]

    输出：
        obs_tensor:
            形状为 (1, T, C, H, W)，dtype=float32，range=[0,1]
    """
    assert obs_sequence.ndim == 4
    assert obs_sequence.shape[-1] == 3
    assert obs_sequence.dtype == np.uint8

    tensor = torch.from_numpy(obs_sequence).float() / 255.0

    # T,H,W,C -> T,C,H,W
    tensor = tensor.permute(0, 3, 1, 2)

    # T,C,H,W -> B,T,C,H,W
    tensor = tensor.unsqueeze(0)

    return tensor.to(device)


def encode_sequence(
    encoder: VisualEncoder,
    obs_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    对图像序列逐帧编码。

    输入：
        obs_tensor:
            形状为 (B, T, C, H, W)

    输出：
        embeddings:
            形状为 (B, T, embedding_dim)
    """
    batch_size, seq_len, channels, height, width = obs_tensor.shape

    flat_obs = obs_tensor.reshape(batch_size * seq_len, channels, height, width)

    flat_embeddings = encoder(flat_obs)

    embeddings = flat_embeddings.reshape(batch_size, seq_len, -1)

    return embeddings


def collect_sequence_with_rewards(env: GridNavEnv, seq_len: int = 12):
    """
    从环境中采集一段序列，同时保存 previous action 和 reward。

    时间步对齐约定：
        obs_0 来自 reset，没有真实 previous action，也没有真实 reward。
        所以：
            prev_actions[0] = ACTION_STAY
            rewards[0] = 0.0

        obs_t 对应：
            prev_actions[t] = 进入 obs_t 前执行的动作
            rewards[t] = 执行该动作后获得的 reward
    """

    obs_list = []
    prev_action_list = []
    reward_list = []
    done_list = []

    obs = env.reset()

    obs_list.append(obs)
    prev_action_list.append(GridNavEnv.ACTION_STAY)
    reward_list.append(0.0)
    done_list.append(False)

    # 这里允许 forward 出现，这样 reward 更有变化。
    all_actions = [
        GridNavEnv.ACTION_FORWARD,
        GridNavEnv.ACTION_TURN_LEFT,
        GridNavEnv.ACTION_TURN_RIGHT,
        GridNavEnv.ACTION_STAY,
    ]

    for _ in range(seq_len - 1):
        action = int(np.random.choice(all_actions))

        obs, reward, done, info = env.step(action)

        obs_list.append(obs)
        prev_action_list.append(action)
        reward_list.append(float(reward))
        done_list.append(bool(done))

        if done:
            break

    # 如果 episode 提前结束，就用最后一帧补齐。
    # 这里只是为了测试 shape，正式训练时 replay buffer 会按 episode 处理。
    while len(obs_list) < seq_len:
        obs_list.append(obs_list[-1])
        prev_action_list.append(GridNavEnv.ACTION_STAY)
        reward_list.append(0.0)
        done_list.append(True)

    obs_sequence = np.stack(obs_list, axis=0)
    prev_actions = np.array(prev_action_list, dtype=np.int64)
    rewards = np.array(reward_list, dtype=np.float32)
    dones = np.array(done_list, dtype=np.bool_)

    return obs_sequence, prev_actions, rewards, dones


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"当前设备: {device}")
    print("=" * 60)

    env = GridNavEnv(
        map_width=10,
        map_height=10,
        image_size=64,
        max_steps=50,
        random_reset=True,
        seed=42,
    )

    seq_len = 12

    obs_sequence, prev_actions, rewards, dones = collect_sequence_with_rewards(
        env=env,
        seq_len=seq_len,
    )

    print("[采集到的环境序列]")
    print(f"obs_sequence shape: {obs_sequence.shape}")
    print(f"prev_actions shape: {prev_actions.shape}")
    print(f"rewards shape: {rewards.shape}")
    print(f"dones shape: {dones.shape}")
    print(f"prev_actions: {prev_actions.tolist()}")
    print(f"rewards: {[round(float(r), 3) for r in rewards.tolist()]}")
    print(f"dones: {dones.tolist()}")

    obs_tensor = obs_sequence_to_tensor(obs_sequence, device)
    actions_tensor = torch.from_numpy(prev_actions).long().unsqueeze(0).to(device)

    # reward target 形状对齐为 (B, T, 1)
    reward_targets = torch.from_numpy(rewards).float().view(1, seq_len, 1).to(device)

    print("\n[转换为模型输入]")
    print(f"obs_tensor shape: {obs_tensor.shape}")
    print(f"actions_tensor shape: {actions_tensor.shape}")
    print(f"reward_targets shape: {reward_targets.shape}")

    encoder = VisualEncoder(
        image_size=64,
        in_channels=3,
        embedding_dim=256,
    ).to(device)

    rssm = RSSM(
        embedding_dim=256,
        num_actions=4,
        action_embed_dim=32,
        deter_dim=256,
        stoch_dim=32,
        hidden_dim=256,
        min_std=0.1,
    ).to(device)

    reward_model = RewardModel(
        feature_dim=288,
        hidden_dim=256,
    ).to(device)

    encoder.eval()
    rssm.eval()
    reward_model.eval()

    print("\n[模型参数量]")
    print(f"Encoder 参数量:      {count_parameters(encoder):,}")
    print(f"RSSM 参数量:         {count_parameters(rssm):,}")
    print(f"RewardModel 参数量:  {count_parameters(reward_model):,}")

    with torch.no_grad():
        embeddings = encode_sequence(encoder, obs_tensor)

        posteriors, priors = rssm.observe(
            embeddings=embeddings,
            actions=actions_tensor,
        )

        posterior_feat = rssm.get_feat(posteriors)

        pred_rewards = reward_model(posterior_feat)

        reward_loss = F.mse_loss(pred_rewards, reward_targets)

    print("\n[中间特征]")
    print(f"embeddings shape:       {embeddings.shape}")
    print(f"posterior_feat shape:   {posterior_feat.shape}")

    print("\n[RewardModel 输出]")
    print(f"pred_rewards shape:     {pred_rewards.shape}")
    print(f"reward_targets shape:   {reward_targets.shape}")
    print(f"reward_loss:            {reward_loss.item():.6f}")

    print("\n[预测 reward 示例]")
    pred_reward_list = pred_rewards.squeeze(0).squeeze(-1).cpu().numpy().tolist()
    target_reward_list = reward_targets.squeeze(0).squeeze(-1).cpu().numpy().tolist()

    for t in range(seq_len):
        print(
            f"t={t:02d}, "
            f"target={target_reward_list[t]: .4f}, "
            f"pred={pred_reward_list[t]: .4f}"
        )

    assert embeddings.shape == (1, seq_len, 256)
    assert posterior_feat.shape == (1, seq_len, 288)
    assert pred_rewards.shape == (1, seq_len, 1)
    assert reward_targets.shape == (1, seq_len, 1)

    print("\n测试通过：Encoder + RSSM + RewardModel 可以正常串联，并且可以计算 reward loss。")


if __name__ == "__main__":
    main()