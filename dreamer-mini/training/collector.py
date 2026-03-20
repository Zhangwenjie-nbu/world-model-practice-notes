import numpy as np

from data.episode import Episode


class RandomPolicy:
    """
    一个最简单的策略：始终输出随机动作。

    为什么先写这个类？
    -----------------
    因为当前我们还没有 actor 网络。
    但 collector 需要一个“策略接口”来决定动作。

    所以这里先定义一个最小策略对象，后面真正的 ActorPolicy
    也可以复用相同的调用方式。
    """

    def __init__(self, env):
        self.env = env

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        根据当前观测返回动作。

        当前版本忽略 obs，直接随机采样。
        这是因为现在只是 warm-up 数据收集阶段。
        """
        return self.env.sample_random_action()


class Collector:
    """
    数据采集器：负责让策略与环境交互，并采集一整条 episode。

    这是训练系统里的一个独立模块，职责非常明确：
        1. reset 环境
        2. 用 policy 决定动作
        3. step 环境
        4. 把 transition 追加到当前 episode
        5. episode 结束后返回整条轨迹

    为什么要把它单独做成类？
    ------------------------
    因为后面会有两种采集场景：
        - 训练采集（可能带探索）
        - 评估采集（通常不用探索）

    提前模块化，后面扩展很方便。
    """

    def __init__(self, env):
        self.env = env

    def collect_episode(self, policy, max_steps: int = 1000) -> Episode:
        """
        采集一整条 episode。

        参数:
            policy:   具有 act(obs) 方法的策略对象
            max_steps: 单条 episode 最多采样多少步，防止异常情况下无限循环

        返回:
            Episode 对象
        """
        obs, _ = self.env.reset()

        # 用 list 动态积累每一步数据。
        # 为什么不用一开始就分配固定长度数组？
        # 因为每条 episode 的长度可能不同，list.append 更自然。
        obs_list = []
        action_list = []
        reward_list = []
        done_list = []
        next_obs_list = []

        for step in range(max_steps):
            # 1) 根据当前观测选动作
            action = policy.act(obs)

            # 2) 与环境交互，拿到下一时刻结果
            next_obs, reward, done, _ = self.env.step(action)

            # 3) 保存这一条 transition: (obs, action, reward, done, next_obs)
            obs_list.append(obs.copy())
            action_list.append(action.copy())
            reward_list.append(reward)
            done_list.append(done)
            next_obs_list.append(next_obs.copy())

            # 4) 更新当前观测，进入下一轮循环
            obs = next_obs

            # 5) 如果 episode 结束，就停止采样
            if done:
                break

        return Episode(
            obs=obs_list,
            actions=action_list,
            rewards=reward_list,
            dones=done_list,
            next_obs=next_obs_list,
        )