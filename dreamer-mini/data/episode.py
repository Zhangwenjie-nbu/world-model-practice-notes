from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Episode:
    """
    一条完整轨迹（episode）的数据容器。

    为什么要单独定义这个类？
    ------------------------
    Dreamer 后面不是直接训练单步 transition，而是训练一段时间序列。
    但在真正做 replay buffer 之前，我们需要先把“单条轨迹的数据结构”定义清楚。

    这里显式保存：
        - obs:      当前时刻观测 o_t
        - actions:  当前时刻动作 a_t
        - rewards:  执行动作后得到的奖励 r_t
        - dones:    执行动作后是否终止 d_t
        - next_obs: 执行动作后到达的下一时刻观测 o_{t+1}

    这样做的好处：
        1. 语义非常直接，便于理解和调试
        2. 后面很容易转换成序列训练所需的格式
        3. 能帮助我们先把“数据对齐关系”理顺
    """
    obs: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]
    next_obs: List[np.ndarray]

    def __len__(self) -> int:
        """
        返回这条 episode 中包含多少个 transition。

        例如：
            如果环境走了 200 步，那么这条 episode 的长度就是 200。
        """
        return len(self.actions)

    def to_numpy(self):
        """
        把内部的 Python list 统一转成 numpy 数组。

        为什么要提供这个函数？
        ----------------------
        因为采样阶段我们用 list.append() 最方便，
        但后面一旦要送入 replay buffer 或 torch tensor，最好统一成 numpy array。

        返回格式：
            obs:      [T, obs_dim]
            actions:  [T, action_dim]
            rewards:  [T]
            dones:    [T]
            next_obs: [T, obs_dim]
        """
        return {
            "obs": np.asarray(self.obs, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "next_obs": np.asarray(self.next_obs, dtype=np.float32),
        }