# memory/replay_buffer.py

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np


@dataclass
class Episode:
    """
    一条 episode 数据。

    对齐约定：
        obs[0] 来自 env.reset()
        actions[0] = ACTION_FORWARD，dummy previous action
        rewards[0] = 0.0

        obs[t] 对应进入该观测前执行的动作 actions[t]
        rewards[t] 是进入 obs[t] 时收到的 reward
    """
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    successes: np.ndarray
    collisions: np.ndarray
    start_path_length: Optional[int] = None


class EpisodeReplayBuffer:
    """
    基于 episode 的 replay buffer。
    """

    def __init__(
        self,
        capacity_episodes: int = 1000,
        padding_action: int = 0,
        success_capacity_episodes: int = 0,
        success_sample_ratio: float = 0.0,
        success_terminal_sample_ratio: float = 0.8,
        prioritize_long_success: bool = True,
        long_success_sampling_alpha: float = 1.0,
    ):
        if capacity_episodes <= 0:
            raise ValueError("capacity_episodes should be positive.")

        if success_capacity_episodes < 0:
            raise ValueError("success_capacity_episodes should be non-negative.")

        if success_capacity_episodes >= capacity_episodes:
            raise ValueError(
                "success_capacity_episodes should be smaller than capacity_episodes."
            )

        if not 0.0 <= success_sample_ratio <= 1.0:
            raise ValueError("success_sample_ratio should be in [0, 1].")

        if not 0.0 <= success_terminal_sample_ratio <= 1.0:
            raise ValueError("success_terminal_sample_ratio should be in [0, 1].")

        if long_success_sampling_alpha < 0.0:
            raise ValueError("long_success_sampling_alpha should be non-negative.")

        self.capacity_episodes = capacity_episodes
        self.padding_action = padding_action
        self.success_capacity_episodes = success_capacity_episodes
        self.success_sample_ratio = success_sample_ratio
        self.success_terminal_sample_ratio = success_terminal_sample_ratio
        self.prioritize_long_success = prioritize_long_success
        self.long_success_sampling_alpha = long_success_sampling_alpha
        self.regular_capacity_episodes = capacity_episodes - success_capacity_episodes
        self._regular_episodes: List[Episode] = []
        self._success_episodes: List[Episode] = []

    @property
    def episodes(self) -> List[Episode]:
        return self._regular_episodes + self._success_episodes

    def __len__(self) -> int:
        return len(self._regular_episodes) + self.num_success_episodes

    @property
    def num_steps(self) -> int:
        return int(sum(len(ep.obs) for ep in self.episodes))

    @property
    def num_success_episodes(self) -> int:
        return len(self._success_episodes)

    @property
    def num_regular_episodes(self) -> int:
        return len(self._regular_episodes)

    @property
    def max_success_path_length(self) -> int:
        if len(self._success_episodes) == 0:
            return -1

        return max(self._success_path_length(ep) for ep in self._success_episodes)

    @staticmethod
    def _success_path_length(ep: Episode) -> int:
        if ep.start_path_length is None:
            return -1

        return int(ep.start_path_length)

    def _drop_low_priority_success_episode(self):
        if len(self._success_episodes) == 0:
            return

        if not self.prioritize_long_success:
            self._success_episodes.pop(0)
            return

        drop_idx = min(
            range(len(self._success_episodes)),
            key=lambda idx: (self._success_path_length(self._success_episodes[idx]), idx),
        )
        self._success_episodes.pop(drop_idx)

    def add_episode(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        successes: np.ndarray = None,
        collisions: np.ndarray = None,
        start_path_length: Optional[int] = None,
    ):
        assert obs.ndim == 4
        assert obs.shape[-1] == 3
        assert obs.dtype == np.uint8

        assert actions.ndim == 1
        assert rewards.ndim == 1
        assert dones.ndim == 1

        assert len(obs) == len(actions) == len(rewards) == len(dones)

        if successes is None:
            successes = np.zeros_like(dones, dtype=np.bool_)

        if collisions is None:
            collisions = np.zeros_like(dones, dtype=np.bool_)

        assert len(obs) == len(successes) == len(collisions)

        episode = Episode(
            obs=obs,
            actions=actions.astype(np.int64),
            rewards=rewards.astype(np.float32),
            dones=dones.astype(np.bool_),
            successes=successes.astype(np.bool_),
            collisions=collisions.astype(np.bool_),
            start_path_length=start_path_length,
        )

        is_success_episode = bool(episode.successes.any())

        if is_success_episode and self.success_capacity_episodes > 0:
            self._success_episodes.append(episode)
            if len(self._success_episodes) > self.success_capacity_episodes:
                self._drop_low_priority_success_episode()
        else:
            self._regular_episodes.append(episode)
            if len(self._regular_episodes) > self.regular_capacity_episodes:
                self._regular_episodes.pop(0)

    def _sample_success_episode(self) -> Episode:
        if len(self._success_episodes) == 0:
            raise RuntimeError("Success replay 为空，无法采样成功样本。")

        if not self.prioritize_long_success or self.long_success_sampling_alpha == 0.0:
            return self._success_episodes[np.random.randint(0, len(self._success_episodes))]

        weights = np.array(
            [max(self._success_path_length(ep), 0) + 1.0 for ep in self._success_episodes],
            dtype=np.float64,
        )
        weights = np.power(weights, self.long_success_sampling_alpha)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            return self._success_episodes[np.random.randint(0, len(self._success_episodes))]

        probs = weights / weight_sum
        index = int(np.random.choice(len(self._success_episodes), p=probs))
        return self._success_episodes[index]

    def _sample_episode(self) -> Episode:
        total_episodes = len(self)
        if total_episodes == 0:
            raise RuntimeError("Replay buffer 为空，无法采样。")

        num_success = self.num_success_episodes
        num_regular = len(self._regular_episodes)

        if num_success == 0:
            return self._regular_episodes[np.random.randint(0, num_regular)]

        if num_regular == 0:
            return self._sample_success_episode()

        base_success_prob = num_success / float(total_episodes)
        success_prob = base_success_prob + (1.0 - base_success_prob) * self.success_sample_ratio

        if np.random.rand() < success_prob:
            return self._sample_success_episode()

        return self._regular_episodes[np.random.randint(0, num_regular)]

    def _sample_sequence_from_episode(self, ep: Episode, seq_len: int) -> Dict[str, np.ndarray]:
        ep_len = len(ep.obs)

        if ep_len >= seq_len:
            success_indices = np.flatnonzero(ep.successes)
            use_success_terminal_window = (
                success_indices.size > 0
                and np.random.rand() < self.success_terminal_sample_ratio
            )

            if use_success_terminal_window:
                terminal_index = int(success_indices[0])
                min_start = max(0, terminal_index - seq_len + 1)
                max_start = min(terminal_index, ep_len - seq_len)

                if max_start >= min_start:
                    start = np.random.randint(min_start, max_start + 1)
                else:
                    start = max(0, min(terminal_index, ep_len - seq_len))
            else:
                start = np.random.randint(0, ep_len - seq_len + 1)

            end = start + seq_len

            obs = ep.obs[start:end]
            actions = ep.actions[start:end]
            rewards = ep.rewards[start:end]
            dones = ep.dones[start:end]
            valid = np.ones(seq_len, dtype=np.float32)
            successes = ep.successes[start:end]
            collisions = ep.collisions[start:end]

        else:
            pad_len = seq_len - ep_len

            obs_pad = np.repeat(ep.obs[-1:], pad_len, axis=0)
            actions_pad = np.full(
                pad_len,
                self.padding_action,
                dtype=np.int64,
            )
            rewards_pad = np.zeros(pad_len, dtype=np.float32)
            dones_pad = np.ones(pad_len, dtype=np.bool_)
            successes_pad = np.zeros(pad_len, dtype=np.bool_)
            collisions_pad = np.zeros(pad_len, dtype=np.bool_)

            obs = np.concatenate([ep.obs, obs_pad], axis=0)
            actions = np.concatenate([ep.actions, actions_pad], axis=0)
            rewards = np.concatenate([ep.rewards, rewards_pad], axis=0)
            dones = np.concatenate([ep.dones, dones_pad], axis=0)
            successes = np.concatenate([ep.successes, successes_pad], axis=0)
            collisions = np.concatenate([ep.collisions, collisions_pad], axis=0)

            valid = np.concatenate(
                [
                    np.ones(ep_len, dtype=np.float32),
                    np.zeros(pad_len, dtype=np.float32),
                ],
                axis=0,
            )

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "successes": successes,
            "collisions": collisions,
            "valid": valid,
        }


    def sample_sequence(self, seq_len: int) -> Dict[str, np.ndarray]:
        ep = self._sample_episode()
        return self._sample_sequence_from_episode(ep, seq_len)

    def sample_success_sequence(self, seq_len: int) -> Dict[str, np.ndarray]:
        ep = self._sample_success_episode()
        return self._sample_sequence_from_episode(ep, seq_len)

    def sample_success_batch(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Dict[str, np.ndarray]:
        if self.num_success_episodes == 0:
            raise RuntimeError("Success replay 为空，无法采样成功 batch。")

        batch = [self.sample_success_sequence(seq_len) for _ in range(batch_size)]

        obs = np.stack([x["obs"] for x in batch], axis=0)
        actions = np.stack([x["actions"] for x in batch], axis=0)
        rewards = np.stack([x["rewards"] for x in batch], axis=0)
        dones = np.stack([x["dones"] for x in batch], axis=0)
        valid = np.stack([x["valid"] for x in batch], axis=0)
        successes = np.stack([x["successes"] for x in batch], axis=0)
        collisions = np.stack([x["collisions"] for x in batch], axis=0)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "successes": successes,
            "collisions": collisions,
            "valid": valid,
        }

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Dict[str, np.ndarray]:
        batch = [self.sample_sequence(seq_len) for _ in range(batch_size)]

        obs = np.stack([x["obs"] for x in batch], axis=0)
        actions = np.stack([x["actions"] for x in batch], axis=0)
        rewards = np.stack([x["rewards"] for x in batch], axis=0)
        dones = np.stack([x["dones"] for x in batch], axis=0)
        valid = np.stack([x["valid"] for x in batch], axis=0)
        successes = np.stack([x["successes"] for x in batch], axis=0)
        collisions = np.stack([x["collisions"] for x in batch], axis=0)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "successes": successes,
            "collisions": collisions,
            "valid": valid,
        }