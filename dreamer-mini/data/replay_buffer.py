import random
from typing import Dict, List

import numpy as np
import torch


class ReplayBuffer:
    """
    用于存储多条 episode，并按固定长度采样时间序列。

    设计目标
    --------
    这是 Dreamer-mini 的第一版 replay buffer，强调“教学清晰”和“便于调试”。

    它的核心职责有两个：
        1. 存储很多条完整 episode
        2. 从某条 episode 中随机裁切长度为 seq_len 的连续子序列

    为什么按 episode 存，而不是平铺成一个大数组？
    ----------------------------------------
    因为 Dreamer 的 world model 训练需要连续时间片段，
    如果直接把所有 transition 打平，就必须额外处理：
        - episode 边界
        - 禁止跨 episode 采样
        - 覆盖旧数据时边界同步维护

    这些问题在第一版教学实现里会增加不必要复杂度。
    所以这里直接按 episode 存，逻辑最清晰。
    """

    def __init__(self, capacity: int, seq_len: int, device: torch.device):
        """
        参数
        ----
        capacity:
            buffer 允许容纳的最大 transition 数量，而不是最大 episode 数量。

        seq_len:
            每次采样的序列长度 T。

        device:
            采样后把 batch tensor 放到哪个设备上。
        """
        self.capacity = capacity
        self.seq_len = seq_len
        self.device = device

        # 按 episode 存储，每个元素都是一个 to_numpy() 之后的 dict
        self.episodes: List[Dict[str, np.ndarray]] = []

        # 当前 buffer 中一共存了多少个 transition
        self.num_steps = 0

    def __len__(self) -> int:
        """
        返回当前 buffer 中的 transition 总数。
        """
        return self.num_steps

    def num_episodes(self) -> int:
        """
        返回当前 buffer 中存储的 episode 条数。
        """
        return len(self.episodes)

    def add_episode(self, episode) -> None:
        """
        向 buffer 中加入一条完整 episode。

        参数
        ----
        episode:
            前一课定义的 Episode 对象。

        处理逻辑
        --------
        1. 先把 episode 转成 numpy 数组，便于后续切片
        2. 如果这条 episode 长度小于 seq_len，则先不加入
           因为它无法提供合法的训练序列
        3. 加入后检查容量，如果超过 capacity，就从最旧 episode 开始删除

        为什么短 episode 直接跳过？
        -------------------------
        因为我们的第一版采样逻辑要求每次都切出完整长度 seq_len 的片段。
        如果 episode 太短，就无法采出合法训练样本。
        """
        data = episode.to_numpy()
        ep_len = len(data["actions"])

        if ep_len < self.seq_len:
            return

        self.episodes.append(data)
        self.num_steps += ep_len

        # 如果超出容量，从最旧 episode 开始弹出
        while self.num_steps > self.capacity and len(self.episodes) > 0:
            removed = self.episodes.pop(0)
            self.num_steps -= len(removed["actions"])

    def can_sample(self, batch_size: int) -> bool:
        """
        判断当前 buffer 是否已经可以采样。

        第一版的简单标准：
            只要存在至少 batch_size 条“可供裁切”的 episode，就允许采样。

        这里的“可供裁切”通过 add_episode 时的长度过滤已经保证了。
        """
        return len(self.episodes) >= batch_size

    def _sample_subsequence_from_episode(self, episode_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        从一条 episode 中随机裁切一个长度为 seq_len 的连续子序列。

        输入:
            episode_data 是一个字典，字段包括：
                obs, actions, rewards, dones, next_obs

        输出:
            仍然是一个字典，但每个字段都只保留长度为 seq_len 的片段。

        例如：
            原 episode 长度为 200，seq_len 为 32
            则可以随机选择 start in [0, 168]
            然后切出 [start : start+32]
        """
        ep_len = len(episode_data["actions"])
        max_start = ep_len - self.seq_len
        start = random.randint(0, max_start)
        end = start + self.seq_len

        return {
            "obs": episode_data["obs"][start:end],
            "actions": episode_data["actions"][start:end],
            "rewards": episode_data["rewards"][start:end],
            "dones": episode_data["dones"][start:end],
            "next_obs": episode_data["next_obs"][start:end],
        }

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        随机采样一个训练 batch。

        输出张量 shape：
            obs:      [B, T, obs_dim]
            actions:  [B, T, action_dim]
            rewards:  [B, T]
            dones:    [B, T]
            next_obs: [B, T, obs_dim]

        为什么返回字典？
        ---------------
        因为后续 world model、loss、trainer 都会按字段访问数据，
        字典接口最清楚，也便于以后扩展更多字段。
        """
        if not self.can_sample(batch_size):
            raise ValueError(
                f"Not enough episodes to sample. "
                f"Have {len(self.episodes)} episodes, need at least {batch_size}."
            )

        # 1. 随机选 batch_size 条 episode（允许重复抽到同一条）
        # 为什么允许重复？
        # 因为这是常见做法，简单直接，而且不会破坏训练正确性。
        sampled_episodes = random.choices(self.episodes, k=batch_size)

        # 2. 从每条 episode 中再随机裁一个长度为 seq_len 的片段
        subsequences = [
            self._sample_subsequence_from_episode(ep)
            for ep in sampled_episodes
        ]

        # 3. 按字段把它们堆叠成 batch
        batch_np = {
            "obs": np.stack([subseq["obs"] for subseq in subsequences], axis=0),
            "actions": np.stack([subseq["actions"] for subseq in subsequences], axis=0),
            "rewards": np.stack([subseq["rewards"] for subseq in subsequences], axis=0),
            "dones": np.stack([subseq["dones"] for subseq in subsequences], axis=0),
            "next_obs": np.stack([subseq["next_obs"] for subseq in subsequences], axis=0),
        }

        # 4. 转成 torch tensor，并放到目标设备上
        batch_torch = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in batch_np.items()
        }

        return batch_torch