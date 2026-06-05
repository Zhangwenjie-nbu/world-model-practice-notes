# scripts/test_rssm.py

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
from models.common import count_parameters


def obs_sequence_to_tensor(obs_sequence: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    将环境图像序列转换为 PyTorch Tensor。

    输入：
        obs_sequence:
            numpy 数组，形状为 (T, H, W, C)，dtype=uint8，range=[0,255]

    输出：
        obs_tensor:
            torch.Tensor，形状为 (1, T, C, H, W)，dtype=float32，range=[0,1]
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

    # 合并 B 和 T，送入 CNN
    flat_obs = obs_tensor.reshape(batch_size * seq_len, channels, height, width)

    flat_embeddings = encoder(flat_obs)

    # 还原成序列形式
    embeddings = flat_embeddings.reshape(batch_size, seq_len, -1)

    return embeddings


def collect_short_sequence(env: GridNavEnv, seq_len: int = 8):
    """
    从环境中采集一段短序列。

    注意：
    actions_prev[t] 表示进入 obs_t 之前执行的动作。

    因此：
    - obs_0 没有真实 previous action，使用 stay 作为 dummy action；
    - obs_1 对应 action_0；
    - obs_2 对应 action_1；
    - ...
    """
    obs_list = []
    prev_action_list = []

    obs = env.reset()

    obs_list.append(obs)
    prev_action_list.append(GridNavEnv.ACTION_STAY)

    # 为了测试稳定，优先采样不会导致碰撞的动作：转向和停留
    # 这里不是为了训练，只是为了验证 RSSM 维度和流程。
    safe_actions = [
        GridNavEnv.ACTION_TURN_LEFT,
        GridNavEnv.ACTION_TURN_RIGHT,
        GridNavEnv.ACTION_STAY,
    ]

    for _ in range(seq_len - 1):
        action = int(np.random.choice(safe_actions))

        obs, reward, done, info = env.step(action)

        obs_list.append(obs)
        prev_action_list.append(action)

        if done:
            break

    # 如果意外提前结束，就补齐序列
    while len(obs_list) < seq_len:
        obs_list.append(obs_list[-1])
        prev_action_list.append(GridNavEnv.ACTION_STAY)

    obs_sequence = np.stack(obs_list, axis=0)
    prev_actions = np.array(prev_action_list, dtype=np.int64)

    return obs_sequence, prev_actions


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

    seq_len = 8
    obs_sequence, prev_actions = collect_short_sequence(env, seq_len=seq_len)

    print("[采集到的环境序列]")
    print(f"obs_sequence shape: {obs_sequence.shape}")
    print(f"obs_sequence dtype: {obs_sequence.dtype}")
    print(f"prev_actions shape: {prev_actions.shape}")
    print(f"prev_actions: {prev_actions.tolist()}")

    obs_tensor = obs_sequence_to_tensor(obs_sequence, device)
    actions_tensor = torch.from_numpy(prev_actions).long().unsqueeze(0).to(device)

    print("\n[转换为模型输入]")
    print(f"obs_tensor shape: {obs_tensor.shape}")
    print(f"actions_tensor shape: {actions_tensor.shape}")

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

    encoder.eval()
    rssm.eval()

    print("\n[模型参数量]")
    print(f"Encoder 参数量: {count_parameters(encoder):,}")
    print(f"RSSM 参数量: {count_parameters(rssm):,}")

    with torch.no_grad():
        embeddings = encode_sequence(encoder, obs_tensor)

        posteriors, priors = rssm.observe(
            embeddings=embeddings,
            actions=actions_tensor,
        )

        posterior_feat = rssm.get_feat(posteriors)
        prior_feat = rssm.get_feat(priors)

    print("\n[Encoder 输出]")
    print(f"embeddings shape: {embeddings.shape}")

    print("\n[RSSM posterior 输出]")
    print(f"posterior.deter shape: {posteriors.deter.shape}")
    print(f"posterior.stoch shape: {posteriors.stoch.shape}")
    print(f"posterior.mean shape:  {posteriors.mean.shape}")
    print(f"posterior.std shape:   {posteriors.std.shape}")
    print(f"posterior_feat shape:  {posterior_feat.shape}")

    print("\n[RSSM prior 输出]")
    print(f"prior.deter shape: {priors.deter.shape}")
    print(f"prior.stoch shape: {priors.stoch.shape}")
    print(f"prior.mean shape:  {priors.mean.shape}")
    print(f"prior.std shape:   {priors.std.shape}")
    print(f"prior_feat shape:  {prior_feat.shape}")

    # 从最后一个 posterior state 出发，测试 imagination rollout
    last_state = rssm.select_state(posteriors, index=-1)

    imagination_horizon = 5
    future_actions = torch.randint(
        low=0,
        high=env.num_actions,
        size=(1, imagination_horizon),
        device=device,
    )

    with torch.no_grad():
        imagined_states = rssm.imagine(
            init_state=last_state,
            actions=future_actions,
        )

        imagined_feat = rssm.get_feat(imagined_states)

    print("\n[Imagination rollout 输出]")
    print(f"future_actions shape: {future_actions.shape}")
    print(f"future_actions: {future_actions.cpu().numpy().tolist()}")
    print(f"imagined.deter shape: {imagined_states.deter.shape}")
    print(f"imagined.stoch shape: {imagined_states.stoch.shape}")
    print(f"imagined_feat shape:  {imagined_feat.shape}")

    assert embeddings.shape == (1, seq_len, 256)
    assert posteriors.deter.shape == (1, seq_len, 256)
    assert posteriors.stoch.shape == (1, seq_len, 32)
    assert posterior_feat.shape == (1, seq_len, 288)
    assert imagined_states.deter.shape == (1, imagination_horizon, 256)
    assert imagined_states.stoch.shape == (1, imagination_horizon, 32)
    assert imagined_feat.shape == (1, imagination_horizon, 288)

    print("\n测试通过：Encoder + RSSM 的 observation rollout 和 imagination rollout 均可正常运行。")


if __name__ == "__main__":
    main()