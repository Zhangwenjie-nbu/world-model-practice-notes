# scripts/test_actor.py

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
from models.actor import DiscreteActor
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


def collect_sequence(env: GridNavEnv, seq_len: int = 10):
    """
    从环境中采集一段序列。

    时间步对齐约定：
        obs_0 来自 reset。
        因此：
            prev_actions[0] = ACTION_STAY

        obs_t 对应：
            prev_actions[t] = 进入 obs_t 前执行的动作
    """

    obs_list = []
    prev_action_list = []

    obs = env.reset()

    obs_list.append(obs)
    prev_action_list.append(GridNavEnv.ACTION_STAY)

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

        if done:
            break

    # 测试阶段为了保持 shape 一致，提前结束时补齐最后一帧。
    while len(obs_list) < seq_len:
        obs_list.append(obs_list[-1])
        prev_action_list.append(GridNavEnv.ACTION_STAY)

    obs_sequence = np.stack(obs_list, axis=0)
    prev_actions = np.array(prev_action_list, dtype=np.int64)

    return obs_sequence, prev_actions


def action_name(action: int) -> str:
    """
    将动作编号转换为可读名称。
    """
    mapping = {
        GridNavEnv.ACTION_FORWARD: "forward",
        GridNavEnv.ACTION_TURN_LEFT: "turn_left",
        GridNavEnv.ACTION_TURN_RIGHT: "turn_right",
        GridNavEnv.ACTION_STAY: "stay",
    }
    return mapping[int(action)]


def main():
    np.random.seed(0)
    torch.manual_seed(0)

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

    seq_len = 10

    obs_sequence, prev_actions = collect_sequence(
        env=env,
        seq_len=seq_len,
    )

    print("[采集到的环境序列]")
    print(f"obs_sequence shape: {obs_sequence.shape}")
    print(f"prev_actions shape: {prev_actions.shape}")
    print(f"prev_actions: {prev_actions.tolist()}")

    obs_tensor = obs_sequence_to_tensor(obs_sequence, device)
    actions_tensor = torch.from_numpy(prev_actions).long().unsqueeze(0).to(device)

    print("\n[转换为模型输入]")
    print(f"obs_tensor shape:      {obs_tensor.shape}")
    print(f"actions_tensor shape:  {actions_tensor.shape}")

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

    actor = DiscreteActor(
        feature_dim=288,
        hidden_dim=256,
        num_actions=4,
    ).to(device)

    encoder.eval()
    rssm.eval()
    actor.eval()

    print("\n[模型参数量]")
    print(f"Encoder 参数量:  {count_parameters(encoder):,}")
    print(f"RSSM 参数量:     {count_parameters(rssm):,}")
    print(f"Actor 参数量:    {count_parameters(actor):,}")

    with torch.no_grad():
        embeddings = encode_sequence(encoder, obs_tensor)

        posteriors, priors = rssm.observe(
            embeddings=embeddings,
            actions=actions_tensor,
        )

        posterior_feat = rssm.get_feat(posteriors)

        # 对整段序列输出动作分布
        action_dist = actor(posterior_feat)

        sampled_actions = action_dist.sample()
        greedy_actions = torch.argmax(action_dist.probs, dim=-1)

        sampled_log_probs = action_dist.log_prob(sampled_actions)
        entropy = action_dist.entropy()

    print("\n[中间特征]")
    print(f"embeddings shape:        {embeddings.shape}")
    print(f"posterior_feat shape:    {posterior_feat.shape}")

    print("\n[Actor 输出]")
    print(f"action probs shape:      {action_dist.probs.shape}")
    print(f"sampled_actions shape:   {sampled_actions.shape}")
    print(f"greedy_actions shape:    {greedy_actions.shape}")
    print(f"log_probs shape:         {sampled_log_probs.shape}")
    print(f"entropy shape:           {entropy.shape}")

    print("\n[动作分布示例]")
    probs_np = action_dist.probs.squeeze(0).cpu().numpy()
    sampled_np = sampled_actions.squeeze(0).cpu().numpy()
    greedy_np = greedy_actions.squeeze(0).cpu().numpy()
    entropy_np = entropy.squeeze(0).cpu().numpy()

    for t in range(seq_len):
        p = probs_np[t]

        print(
            f"t={t:02d}, "
            f"probs=[forward={p[0]:.3f}, left={p[1]:.3f}, right={p[2]:.3f}, stay={p[3]:.3f}], "
            f"sampled={sampled_np[t]}({action_name(sampled_np[t])}), "
            f"greedy={greedy_np[t]}({action_name(greedy_np[t])}), "
            f"entropy={entropy_np[t]:.3f}"
        )

    # 测试单步动作选择接口
    last_feat = posterior_feat[:, -1]

    with torch.no_grad():
        stochastic_action, stochastic_log_prob, stochastic_entropy = actor.sample_action(
            last_feat,
            deterministic=False,
            temperature=1.0,
        )

        deterministic_action, deterministic_log_prob, deterministic_entropy = actor.sample_action(
            last_feat,
            deterministic=True,
            temperature=1.0,
        )

    print("\n[单步动作选择]")
    print(
        f"随机采样动作: {stochastic_action.item()} "
        f"({action_name(stochastic_action.item())}), "
        f"log_prob={stochastic_log_prob.item():.4f}, "
        f"entropy={stochastic_entropy.item():.4f}"
    )

    print(
        f"贪心选择动作: {deterministic_action.item()} "
        f"({action_name(deterministic_action.item())}), "
        f"log_prob={deterministic_log_prob.item():.4f}, "
        f"entropy={deterministic_entropy.item():.4f}"
    )

    assert embeddings.shape == (1, seq_len, 256)
    assert posterior_feat.shape == (1, seq_len, 288)
    assert action_dist.probs.shape == (1, seq_len, env.num_actions)
    assert sampled_actions.shape == (1, seq_len)
    assert greedy_actions.shape == (1, seq_len)
    assert sampled_log_probs.shape == (1, seq_len)
    assert entropy.shape == (1, seq_len)
    assert stochastic_action.shape == (1,)
    assert deterministic_action.shape == (1,)

    print("\n测试通过：Encoder + RSSM + Actor 可以正常串联，并且可以输出动作分布和动作采样结果。")


if __name__ == "__main__":
    main()