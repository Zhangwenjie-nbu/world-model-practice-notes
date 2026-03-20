import gymnasium as gym
import numpy as np


class EnvWrapper:
    """
    对 Gymnasium 环境做一个轻量封装。

    为什么要自己包一层，而不是直接在 train.py 里调 gym？
    ----------------------------------------------
    因为后面训练系统会越来越复杂：
        - collector 要调环境
        - eval 也要调环境
        - 以后可能还会加 action clip、obs 预处理、像素包装器等

    所以最好一开始就统一接口，避免环境逻辑散落在各个文件中。
    """

    def __init__(self, env_name: str):
        self.env = gym.make(env_name)

        # 观测维度，例如 Pendulum-v1 的 obs_dim = 3
        self.obs_dim = int(np.prod(self.env.observation_space.shape))

        # 动作维度，例如 Pendulum-v1 的 action_dim = 1
        self.action_dim = int(np.prod(self.env.action_space.shape))

        # 动作上下界，后面 actor 输出动作时要与环境范围对齐
        self.action_low = self.env.action_space.low.astype(np.float32)
        self.action_high = self.env.action_space.high.astype(np.float32)

    def reset(self):
        """
        重置环境，返回初始观测。

        Gymnasium 的 reset() 返回:
            obs, info

        我们这里只保留一致的 float32 类型，方便后续直接送入网络。
        """
        obs, info = self.env.reset()
        return obs.astype(np.float32), info

    def step(self, action):
        """
        执行一步环境交互。

        输入:
            action: numpy array，形状通常是 [action_dim]

        返回:
            next_obs:  下一时刻观测
            reward:    标量奖励
            done:      是否终止（terminated 或 truncated）
            info:      额外信息
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs.astype(np.float32), float(reward), done, info

    def sample_random_action(self):
        """
        从环境动作空间中随机采样一个动作。

        为什么需要这个函数？
        -------------------
        Dreamer 在训练初期通常需要先收集一些随机数据，
        否则 replay buffer 为空，world model 没法开始训练。

        这里把“随机动作采样”也统一封装进环境类中，
        避免外部直接依赖 gym 原始接口。
        """
        action = self.env.action_space.sample()
        return np.asarray(action, dtype=np.float32)