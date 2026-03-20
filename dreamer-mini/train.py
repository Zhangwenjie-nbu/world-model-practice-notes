import yaml
import torch

from utils.seed import set_seed
from envs.make_env import EnvWrapper
from training.collector import Collector, RandomPolicy
from data.replay_buffer import ReplayBuffer


def load_config(path: str):
    """
    读取 YAML 配置文件。
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # 1. 读取配置并设置随机种子
    cfg = load_config("configs/pendulum.yaml")
    set_seed(cfg["seed"])

    # 2. 选择运行设备
    device = torch.device(
        "cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )

    # 3. 创建环境
    env = EnvWrapper(cfg["env"]["name"])

    # 4. 把环境维度写回配置
    cfg["model"]["obs_dim"] = env.obs_dim
    cfg["model"]["action_dim"] = env.action_dim

    print("Env:", cfg["env"]["name"])
    print("Obs dim:", env.obs_dim)
    print("Action dim:", env.action_dim)
    print("Device:", device)

    # 5. 创建随机策略、采集器、replay buffer
    policy = RandomPolicy(env)
    collector = Collector(env)
    replay_buffer = ReplayBuffer(
        capacity=cfg["buffer"]["capacity"],
        seq_len=cfg["train"]["seq_len"],
        device=device,
    )

    # 6. 先采集若干条 episode 放入 buffer
    num_collect_episodes = 5
    for i in range(num_collect_episodes):
        episode = collector.collect_episode(policy, max_steps=300)
        replay_buffer.add_episode(episode)
        print(f"\nCollected episode {i+1}")
        print("Episode length:", len(episode))
        print("Buffer episodes:", replay_buffer.num_episodes())
        print("Buffer steps:", len(replay_buffer))

    # 7. 判断是否可以采样
    batch_size = 4
    print("\nCan sample:", replay_buffer.can_sample(batch_size))

    if replay_buffer.can_sample(batch_size):
        batch = replay_buffer.sample_batch(batch_size)

        print("\nSampled batch:")
        print("obs shape:     ", tuple(batch["obs"].shape))
        print("actions shape: ", tuple(batch["actions"].shape))
        print("rewards shape: ", tuple(batch["rewards"].shape))
        print("dones shape:   ", tuple(batch["dones"].shape))
        print("next_obs shape:", tuple(batch["next_obs"].shape))

        # 8. 打印一个样本序列中的第 0 个时间步，帮助确认数据形式
        print("\nExample from batch[0, 0]:")
        print("obs[0,0]      =", batch["obs"][0, 0])
        print("action[0,0]   =", batch["actions"][0, 0])
        print("reward[0,0]   =", batch["rewards"][0, 0])
        print("done[0,0]     =", batch["dones"][0, 0])
        print("next_obs[0,0] =", batch["next_obs"][0, 0])


if __name__ == "__main__":
    main()