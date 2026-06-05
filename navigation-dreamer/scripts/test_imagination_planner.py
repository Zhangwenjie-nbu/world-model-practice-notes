# scripts/test_imagination_planner.py

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
from models.reward_model import RewardModel
from models.actor import DiscreteActor
from models.critic import Critic
from models.planner import (
    actor_imagination_rollout,
    select_action_by_random_shooting,
)
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
    从环境中采集一段真实观测序列。

    时间步对齐：
        obs_0 来自 reset。
        prev_actions[0] = ACTION_STAY

        obs_t 对应：
        prev_actions[t] = 进入 obs_t 前执行的动作。
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


def action_name(action: int) -> str:
    """
    将动作编号转换为可读动作名。
    """
    mapping = {
        GridNavEnv.ACTION_FORWARD: "forward",
        GridNavEnv.ACTION_TURN_LEFT: "turn_left",
        GridNavEnv.ACTION_TURN_RIGHT: "turn_right",
        GridNavEnv.ACTION_STAY: "stay",
    }
    return mapping[int(action)]


def format_action_sequence(actions) -> str:
    """
    将动作序列格式化为可读字符串。
    """
    return "[" + ", ".join(action_name(int(a)) for a in actions) + "]"


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
    horizon = 8
    gamma = 0.99
    num_candidates = 64

    obs_sequence, prev_actions, rewards, dones = collect_sequence(
        env=env,
        seq_len=seq_len,
    )

    print("[真实环境序列]")
    print(f"obs_sequence shape: {obs_sequence.shape}")
    print(f"prev_actions: {prev_actions.tolist()}")
    print(f"rewards: {[round(float(r), 3) for r in rewards.tolist()]}")
    print(f"dones: {dones.tolist()}")

    obs_tensor = obs_sequence_to_tensor(obs_sequence, device)
    actions_tensor = torch.from_numpy(prev_actions).long().unsqueeze(0).to(device)

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

    actor = DiscreteActor(
        feature_dim=288,
        hidden_dim=256,
        num_actions=4,
    ).to(device)

    critic = Critic(
        feature_dim=288,
        hidden_dim=256,
    ).to(device)

    encoder.eval()
    rssm.eval()
    reward_model.eval()
    actor.eval()
    critic.eval()

    print("\n[模型参数量]")
    print(f"Encoder 参数量:      {count_parameters(encoder):,}")
    print(f"RSSM 参数量:         {count_parameters(rssm):,}")
    print(f"RewardModel 参数量:  {count_parameters(reward_model):,}")
    print(f"Actor 参数量:        {count_parameters(actor):,}")
    print(f"Critic 参数量:       {count_parameters(critic):,}")

    with torch.no_grad():
        embeddings = encode_sequence(encoder, obs_tensor)

        posteriors, priors = rssm.observe(
            embeddings=embeddings,
            actions=actions_tensor,
        )

        posterior_feat = rssm.get_feat(posteriors)

        # 当前状态取真实观测序列最后一个 posterior state
        current_state = rssm.select_state(posteriors, index=-1)
        current_feat = rssm.get_feat(current_state)

        current_action_dist = actor(current_feat)
        current_greedy_action = torch.argmax(current_action_dist.probs, dim=-1)

    print("\n[当前 latent state]")
    print(f"embeddings shape:        {embeddings.shape}")
    print(f"posterior_feat shape:    {posterior_feat.shape}")
    print(f"current_feat shape:      {current_feat.shape}")

    probs = current_action_dist.probs.squeeze(0).cpu().numpy()
    print("\n[当前 Actor 动作分布]")
    print(
        f"forward={probs[0]:.3f}, "
        f"turn_left={probs[1]:.3f}, "
        f"turn_right={probs[2]:.3f}, "
        f"stay={probs[3]:.3f}"
    )
    print(
        f"Actor greedy action: {current_greedy_action.item()} "
        f"({action_name(current_greedy_action.item())})"
    )

    # ------------------------------------------------------------
    # 1. Actor-driven imagination rollout
    # ------------------------------------------------------------
    with torch.no_grad():
        actor_rollout = actor_imagination_rollout(
            rssm=rssm,
            actor=actor,
            reward_model=reward_model,
            critic=critic,
            init_state=current_state,
            horizon=horizon,
            gamma=gamma,
            deterministic=False,
            temperature=1.0,
        )

    print("\n" + "=" * 60)
    print("[方法 1] Actor-driven imagination rollout")
    print("=" * 60)

    print(f"imagined actions shape:   {actor_rollout.actions.shape}")
    print(f"imagined rewards shape:   {actor_rollout.rewards.shape}")
    print(f"imagined values shape:    {actor_rollout.values.shape}")
    print(f"imagined returns shape:   {actor_rollout.returns.shape}")
    print(f"log_probs shape:          {actor_rollout.log_probs.shape}")
    print(f"entropies shape:          {actor_rollout.entropies.shape}")
    print(f"terminal_value shape:     {actor_rollout.terminal_value.shape}")

    actor_actions_np = actor_rollout.actions.squeeze(0).cpu().numpy()
    actor_rewards_np = actor_rollout.rewards.squeeze(0).squeeze(-1).cpu().numpy()
    actor_values_np = actor_rollout.values.squeeze(0).squeeze(-1).cpu().numpy()
    actor_returns_np = actor_rollout.returns.squeeze(0).squeeze(-1).cpu().numpy()
    actor_entropy_np = actor_rollout.entropies.squeeze(0).cpu().numpy()

    print("\n[Actor imagined trajectory]")
    for t in range(horizon):
        print(
            f"k={t + 1:02d}, "
            f"action={int(actor_actions_np[t])}({action_name(actor_actions_np[t])}), "
            f"imagined_reward={actor_rewards_np[t]: .4f}, "
            f"value={actor_values_np[t]: .4f}, "
            f"return={actor_returns_np[t]: .4f}, "
            f"entropy={actor_entropy_np[t]: .4f}"
        )

    # ------------------------------------------------------------
    # 2. Random shooting action selection
    # ------------------------------------------------------------
    with torch.no_grad():
        shooting_eval = select_action_by_random_shooting(
            rssm=rssm,
            reward_model=reward_model,
            critic=critic,
            init_state=current_state,
            num_actions=env.num_actions,
            horizon=horizon,
            num_candidates=num_candidates,
            gamma=gamma,
        )

    print("\n" + "=" * 60)
    print("[方法 2] Random shooting action selection")
    print("=" * 60)

    print(f"candidate_actions shape: {shooting_eval.candidate_actions.shape}")
    print(f"rewards shape:           {shooting_eval.rewards.shape}")
    print(f"terminal_values shape:   {shooting_eval.terminal_values.shape}")
    print(f"returns shape:           {shooting_eval.returns.shape}")
    print(f"sequence_scores shape:   {shooting_eval.sequence_scores.shape}")

    best_idx = shooting_eval.best_index.item()
    best_actions = shooting_eval.best_actions.cpu().numpy()
    best_action = shooting_eval.best_action.item()
    best_score = shooting_eval.sequence_scores[best_idx].item()

    print("\n[Random shooting 最优候选]")
    print(f"best_index:   {best_idx}")
    print(f"best_score:   {best_score:.6f}")
    print(f"best_actions: {format_action_sequence(best_actions)}")
    print(f"selected first action: {best_action} ({action_name(best_action)})")

    # 打印 top-k 候选序列
    top_k = 5
    top_scores, top_indices = torch.topk(
        shooting_eval.sequence_scores,
        k=min(top_k, num_candidates),
        largest=True,
    )

    print(f"\n[Top-{top_k} candidate sequences]")
    for rank in range(top_indices.shape[0]):
        idx = top_indices[rank].item()
        score = top_scores[rank].item()
        action_seq = shooting_eval.candidate_actions[idx].cpu().numpy()

        print(
            f"rank={rank + 1}, "
            f"index={idx}, "
            f"score={score:.6f}, "
            f"actions={format_action_sequence(action_seq)}"
        )

    # ------------------------------------------------------------
    # shape 检查
    # ------------------------------------------------------------
    assert embeddings.shape == (1, seq_len, 256)
    assert posterior_feat.shape == (1, seq_len, 288)
    assert current_feat.shape == (1, 288)

    assert actor_rollout.actions.shape == (1, horizon)
    assert actor_rollout.rewards.shape == (1, horizon, 1)
    assert actor_rollout.values.shape == (1, horizon, 1)
    assert actor_rollout.returns.shape == (1, horizon, 1)
    assert actor_rollout.log_probs.shape == (1, horizon)
    assert actor_rollout.entropies.shape == (1, horizon)
    assert actor_rollout.terminal_value.shape == (1, 1)

    assert shooting_eval.candidate_actions.shape == (num_candidates, horizon)
    assert shooting_eval.rewards.shape == (num_candidates, horizon, 1)
    assert shooting_eval.terminal_values.shape == (num_candidates, 1)
    assert shooting_eval.returns.shape == (num_candidates, horizon, 1)
    assert shooting_eval.sequence_scores.shape == (num_candidates,)

    print("\n测试通过：Imagination rollout 与 action selection 链路可以正常运行。")


if __name__ == "__main__":
    main()