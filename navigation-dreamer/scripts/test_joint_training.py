# scripts/test_joint_training.py

import sys
from pathlib import Path

import numpy as np
import torch

# 让脚本可以从项目根目录导入模块
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from envs.grid_nav_env import GridNavEnv
from models.encoder import VisualEncoder
from models.rssm import RSSM
from models.decoder import VisualDecoder
from models.reward_model import RewardModel
from models.actor import DiscreteActor
from models.critic import Critic
from models.common import count_parameters
from trainers.joint_trainer import (
    train_world_model_step,
    train_actor_critic_step,
)


def collect_one_sequence(env: GridNavEnv, seq_len: int):
    """
    从环境中采集一条序列。

    时间步对齐：
        obs_0 来自 reset：
            prev_actions[0] = ACTION_STAY
            rewards[0] = 0.0

        obs_t 对应：
            prev_actions[t] = 进入 obs_t 前执行的动作
            rewards[t] = 执行该动作后获得的 reward

    valid_mask:
        1 表示真实采样步；
        0 表示 done 后为了补齐长度的 padding。
    """

    obs_list = []
    prev_action_list = []
    reward_list = []
    done_list = []
    valid_list = []

    obs = env.reset()

    obs_list.append(obs)
    prev_action_list.append(GridNavEnv.ACTION_STAY)
    reward_list.append(0.0)
    done_list.append(False)
    valid_list.append(1.0)

    for _ in range(seq_len - 1):
        action = np.random.randint(0, env.num_actions)

        obs, reward, done, info = env.step(int(action))

        obs_list.append(obs)
        prev_action_list.append(int(action))
        reward_list.append(float(reward))
        done_list.append(bool(done))
        valid_list.append(1.0)

        if done:
            break

    while len(obs_list) < seq_len:
        obs_list.append(obs_list[-1])
        prev_action_list.append(GridNavEnv.ACTION_STAY)
        reward_list.append(0.0)
        done_list.append(True)
        valid_list.append(0.0)

    obs_sequence = np.stack(obs_list, axis=0)
    prev_actions = np.array(prev_action_list, dtype=np.int64)
    rewards = np.array(reward_list, dtype=np.float32)
    dones = np.array(done_list, dtype=np.bool_)
    valid_mask = np.array(valid_list, dtype=np.float32)

    return obs_sequence, prev_actions, rewards, dones, valid_mask


def collect_batch(batch_size: int, seq_len: int, seed: int = 0):
    """
    采集一个 batch 的随机交互序列。
    """

    env = GridNavEnv(
        map_width=10,
        map_height=10,
        image_size=64,
        max_steps=50,
        random_reset=True,
        seed=seed,
    )

    obs_batch = []
    action_batch = []
    reward_batch = []
    done_batch = []
    valid_batch = []

    for _ in range(batch_size):
        obs_seq, actions, rewards, dones, valid = collect_one_sequence(
            env=env,
            seq_len=seq_len,
        )

        obs_batch.append(obs_seq)
        action_batch.append(actions)
        reward_batch.append(rewards)
        done_batch.append(dones)
        valid_batch.append(valid)

    obs_batch = np.stack(obs_batch, axis=0)
    action_batch = np.stack(action_batch, axis=0)
    reward_batch = np.stack(reward_batch, axis=0)
    done_batch = np.stack(done_batch, axis=0)
    valid_batch = np.stack(valid_batch, axis=0)

    return obs_batch, action_batch, reward_batch, done_batch, valid_batch


def batch_to_tensors(
    obs_batch: np.ndarray,
    action_batch: np.ndarray,
    reward_batch: np.ndarray,
    valid_batch: np.ndarray,
    device: torch.device,
):
    """
    将 numpy batch 转换为 PyTorch Tensor。
    """

    # obs_batch: B,T,H,W,C -> B,T,C,H,W
    obs_tensor = torch.from_numpy(obs_batch).float() / 255.0
    obs_tensor = obs_tensor.permute(0, 1, 4, 2, 3).contiguous().to(device)

    actions_tensor = torch.from_numpy(action_batch).long().to(device)

    reward_targets = torch.from_numpy(reward_batch).float().unsqueeze(-1).to(device)

    valid_mask = torch.from_numpy(valid_batch).float().unsqueeze(-1).to(device)

    return obs_tensor, actions_tensor, reward_targets, valid_mask


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"当前设备: {device}")
    print("=" * 60)

    batch_size = 4
    seq_len = 16
    horizon = 8

    obs_batch, action_batch, reward_batch, done_batch, valid_batch = collect_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        seed=42,
    )

    print("[采集到的 batch]")
    print(f"obs_batch shape:     {obs_batch.shape}")
    print(f"action_batch shape:  {action_batch.shape}")
    print(f"reward_batch shape:  {reward_batch.shape}")
    print(f"done_batch shape:    {done_batch.shape}")
    print(f"valid_batch shape:   {valid_batch.shape}")
    print(f"reward 示例:         {[round(float(x), 3) for x in reward_batch[0].tolist()]}")
    print(f"valid 示例:          {valid_batch[0].tolist()}")

    obs_tensor, actions_tensor, reward_targets, valid_mask = batch_to_tensors(
        obs_batch=obs_batch,
        action_batch=action_batch,
        reward_batch=reward_batch,
        valid_batch=valid_batch,
        device=device,
    )

    print("\n[Tensor 输入]")
    print(f"obs_tensor shape:      {obs_tensor.shape}")
    print(f"actions_tensor shape:  {actions_tensor.shape}")
    print(f"reward_targets shape:  {reward_targets.shape}")
    print(f"valid_mask shape:      {valid_mask.shape}")

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

    decoder = VisualDecoder(
        feature_dim=288,
        image_size=64,
        out_channels=3,
        hidden_channels=256,
    ).to(device)

    reward_model = RewardModel(
        feature_dim=288,
        hidden_dim=256,
    ).to(device)

    actor = DiscreteActor(
        feature_dim=288,
        hidden_dim=256,
        num_actions=4,
    ).to(device)

    critic = Critic(
        feature_dim=288,
        hidden_dim=256,
    ).to(device)

    print("\n[模型参数量]")
    print(f"Encoder 参数量:      {count_parameters(encoder):,}")
    print(f"RSSM 参数量:         {count_parameters(rssm):,}")
    print(f"Decoder 参数量:      {count_parameters(decoder):,}")
    print(f"RewardModel 参数量:  {count_parameters(reward_model):,}")
    print(f"Actor 参数量:        {count_parameters(actor):,}")
    print(f"Critic 参数量:       {count_parameters(critic):,}")

    world_model_optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(rssm.parameters())
        + list(decoder.parameters())
        + list(reward_model.parameters()),
        lr=3e-4,
    )

    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=3e-4,
    )

    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=3e-4,
    )

    print("\n" + "=" * 60)
    print("[Step 1] World Model 更新")
    print("=" * 60)

    wm_loss = train_world_model_step(
        encoder=encoder,
        rssm=rssm,
        decoder=decoder,
        reward_model=reward_model,
        optimizer=world_model_optimizer,
        obs_tensor=obs_tensor,
        actions_tensor=actions_tensor,
        reward_targets=reward_targets,
        valid_mask=valid_mask,
        recon_scale=1.0,
        reward_scale=1.0,
        kl_scale=0.1,
        free_nats=0.0,
        grad_clip=100.0,
    )

    print(f"world_model total_loss: {wm_loss.total_loss.item():.6f}")
    print(f"recon_loss:             {wm_loss.recon_loss.item():.6f}")
    print(f"reward_loss:            {wm_loss.reward_loss.item():.6f}")
    print(f"kl_loss:                {wm_loss.kl_loss.item():.6f}")
    print(f"grad_norm:              {wm_loss.grad_norm:.6f}")
    print(f"reconstructed shape:    {wm_loss.reconstructed.shape}")
    print(f"pred_rewards shape:     {wm_loss.pred_rewards.shape}")
    print(f"posterior_feat shape:   {wm_loss.posterior_feat.shape}")

    # 从每条序列最后一个时间步取 posterior state，作为 actor-critic imagination 起点。
    init_state = rssm.select_state(
        wm_loss.posteriors,
        index=-1,
    )

    print("\n" + "=" * 60)
    print("[Step 2] Actor-Critic 更新")
    print("=" * 60)

    ac_loss = train_actor_critic_step(
        rssm=rssm,
        actor=actor,
        reward_model=reward_model,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        init_state=init_state,
        horizon=horizon,
        gamma=0.99,
        entropy_scale=1e-3,
        critic_scale=1.0,
        grad_clip=100.0,
    )

    print(f"actor_critic total_loss: {ac_loss.total_loss.item():.6f}")
    print(f"actor_loss:              {ac_loss.actor_loss.item():.6f}")
    print(f"critic_loss:             {ac_loss.critic_loss.item():.6f}")
    print(f"entropy_loss:            {ac_loss.entropy_loss.item():.6f}")
    print(f"mean_return:             {ac_loss.mean_return.item():.6f}")
    print(f"mean_reward:             {ac_loss.mean_reward.item():.6f}")
    print(f"mean_value:              {ac_loss.mean_value.item():.6f}")
    print(f"actor grad_norm:         {ac_loss.grad_norm_actor:.6f}")
    print(f"critic grad_norm:        {ac_loss.grad_norm_critic:.6f}")

    assert wm_loss.reconstructed.shape == (batch_size, seq_len, 3, 64, 64)
    assert wm_loss.pred_rewards.shape == (batch_size, seq_len, 1)
    assert wm_loss.posterior_feat.shape == (batch_size, seq_len, 288)

    print("\n测试通过：World Model 与 Actor-Critic 均完成了一次可执行训练更新。")


if __name__ == "__main__":
    main()