# memory/bc_buffer.py

from dataclasses import dataclass
from typing import List, Dict

import numpy as np


@dataclass
class BCEpisode:
    """
    专门用于 BC / DAgger 的 episode。

    obs[t]:
        当前观测。

    prev_actions[t]:
        进入 obs[t] 之前真实执行的动作，用于 RSSM observe。

    expert_actions[t]:
        在 obs[t] 这个状态下，专家建议下一步执行的动作。

    valid[t]:
        当前时间步是否有效。
    """
    obs: np.ndarray
    prev_actions: np.ndarray
    expert_actions: np.ndarray
    valid: np.ndarray


class BCReplayBuffer:
    """
    专门用于 BC / DAgger 的 Replay Buffer。

    它和 EpisodeReplayBuffer 的区别是：
    - EpisodeReplayBuffer 存真实动作、reward，用于 World Model；
    - BCReplayBuffer 存专家标签 expert_actions，用于训练 Actor。
    """

    def __init__(self, capacity_episodes: int = 1000, padding_action: int = 3):
        self.capacity_episodes = capacity_episodes
        self.padding_action = padding_action
        self.episodes: List[BCEpisode] = []

    def __len__(self):
        return len(self.episodes)

    @property
    def num_steps(self):
        return int(sum(len(ep.obs) for ep in self.episodes))

    def add_episode(
        self,
        obs: np.ndarray,
        prev_actions: np.ndarray,
        expert_actions: np.ndarray,
        valid: np.ndarray,
    ):
        assert obs.ndim == 4
        assert obs.shape[-1] == 3
        assert len(obs) == len(prev_actions) == len(expert_actions) == len(valid)

        episode = BCEpisode(
            obs=obs.astype(np.uint8),
            prev_actions=prev_actions.astype(np.int64),
            expert_actions=expert_actions.astype(np.int64),
            valid=valid.astype(np.float32),
        )

        self.episodes.append(episode)

        if len(self.episodes) > self.capacity_episodes:
            self.episodes.pop(0)

    def sample_sequence(self, seq_len: int) -> Dict[str, np.ndarray]:
        if len(self.episodes) == 0:
            raise RuntimeError("BCReplayBuffer 为空，无法采样。")

        ep = self.episodes[np.random.randint(0, len(self.episodes))]
        ep_len = len(ep.obs)

        if ep_len >= seq_len:
            start = np.random.randint(0, ep_len - seq_len + 1)
            end = start + seq_len

            obs = ep.obs[start:end]
            prev_actions = ep.prev_actions[start:end]
            expert_actions = ep.expert_actions[start:end]
            valid = ep.valid[start:end]

        else:
            pad_len = seq_len - ep_len

            obs_pad = np.repeat(ep.obs[-1:], pad_len, axis=0)

            prev_actions_pad = np.full(
                pad_len,
                self.padding_action,
                dtype=np.int64,
            )

            expert_actions_pad = np.full(
                pad_len,
                self.padding_action,
                dtype=np.int64,
            )

            valid_pad = np.zeros(pad_len, dtype=np.float32)

            obs = np.concatenate([ep.obs, obs_pad], axis=0)
            prev_actions = np.concatenate([ep.prev_actions, prev_actions_pad], axis=0)
            expert_actions = np.concatenate([ep.expert_actions, expert_actions_pad], axis=0)
            valid = np.concatenate([ep.valid, valid_pad], axis=0)

        return {
            "obs": obs,
            "prev_actions": prev_actions,
            "expert_actions": expert_actions,
            "valid": valid,
        }

    def sample_batch(self, batch_size: int, seq_len: int) -> Dict[str, np.ndarray]:
        batch = [self.sample_sequence(seq_len) for _ in range(batch_size)]

        return {
            "obs": np.stack([x["obs"] for x in batch], axis=0),
            "prev_actions": np.stack([x["prev_actions"] for x in batch], axis=0),
            "expert_actions": np.stack([x["expert_actions"] for x in batch], axis=0),
            "valid": np.stack([x["valid"] for x in batch], axis=0),
        }