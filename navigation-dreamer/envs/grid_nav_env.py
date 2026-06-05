# envs/grid_nav_env.py

import random
from collections import deque
from typing import Tuple, Set, Optional

import numpy as np

from envs.renderer import GridRenderer


class GridNavEnv:
    """
    简单 2D 视觉导航 / 避障环境。

    当前版本修改重点：
    1. 动作空间删除 STAY，仅保留 forward / turn_left / turn_right；
    2. 标准 WM-RL 不再使用 hidden counter 形式的额外 reward / termination；
    3. 奖励只依赖当前可观测状态、当前动作和碰撞/成功事件，降低世界模型学习难度；
    4. 提供 terminate_on_collision 开关，标准 WM-RL 训练可令碰撞只惩罚但不立即终止；
    5. 文件中仍保留 BFS heuristic 作为可选评估/对照工具，但标准训练脚本不使用专家标签。
    """

    ACTION_FORWARD = 0
    ACTION_TURN_LEFT = 1
    ACTION_TURN_RIGHT = 2

    # 方向定义：
    # 0 = up, 1 = right, 2 = down, 3 = left
    DIR_TO_DELTA = {
        0: (0, -1),
        1: (1, 0),
        2: (0, 1),
        3: (-1, 0),
    }

    def __init__(
        self,
        map_width: int = 10,
        map_height: int = 10,
        image_size: int = 64,
        max_steps: int = 100,
        goal_reward: float = 20.0,
        collision_penalty: float = 10.0,
        turn_penalty: float = 0.0,
        progress_reward_scale: float = 1,
        turn_shaping_scale: float = 0.5,
        step_penalty: float = 0.0,
        goal_radius: float = 0.0,
        random_reset: bool = True,
        terminate_on_collision: bool = True,
        reset_min_path_length: Optional[int] = None,
        reset_max_path_length: Optional[int] = None,
        reset_sample_attempts: int = 200,
        seed: int = 0,
    ):
        self.map_width = map_width
        self.map_height = map_height
        self.image_size = image_size
        self.max_steps = max_steps

        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty
        self.turn_penalty = turn_penalty
        self.progress_reward_scale = progress_reward_scale
        self.turn_shaping_scale = turn_shaping_scale
        self.step_penalty = step_penalty
        self.goal_radius = goal_radius
        self.random_reset = random_reset
        self.terminate_on_collision = terminate_on_collision
        self.reset_min_path_length = reset_min_path_length
        self.reset_max_path_length = reset_max_path_length
        self.reset_sample_attempts = reset_sample_attempts

        self.num_actions = 3

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.renderer = GridRenderer(
            map_width=map_width,
            map_height=map_height,
            image_size=image_size,
        )

        self.agent_pos = None
        self.agent_dir = None
        self.goal_pos = None
        self.obstacles = set()
        self.step_count = 0
        self.steps_since_move = 0

        self._init_obstacles()

    def _init_obstacles(self):
        """
        固定障碍物布局。
        """
        obstacles = {
            (3, 3), (3, 4), (3, 5),
            (6, 2), (6, 3), (6, 4),
            (5, 7), (6, 7), (7, 7),
        }

        self.obstacles = {
            (x, y)
            for (x, y) in obstacles
            if 0 <= x < self.map_width and 0 <= y < self.map_height
        }

    def reset(self):
        self.step_count = 0
        self.steps_since_move = 0

        if self.random_reset:
            if (
                self.reset_min_path_length is not None
                or self.reset_max_path_length is not None
            ):
                self.agent_pos, self.goal_pos = self._sample_agent_goal_pair_with_path_constraints()
            else:
                self.agent_pos = self._sample_empty_cell()
                self.goal_pos = self._sample_empty_cell(exclude={self.agent_pos})

                min_dist = max(3.0, min(self.map_width, self.map_height) / 3.0)
                retry_count = 0

                while self._distance(self.agent_pos, self.goal_pos) < min_dist:
                    self.goal_pos = self._sample_empty_cell(exclude={self.agent_pos})
                    retry_count += 1

                    if retry_count > 100:
                        break
        else:
            self.agent_pos = (1, 1)
            self.goal_pos = (self.map_width - 2, self.map_height - 2)

        self.agent_dir = self.rng.choice([0, 1, 2, 3])

        return self.render()

    def step(self, action: int):
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}")

        old_pos = self.agent_pos
        old_dir = self.agent_dir
        old_dist = self._distance(old_pos, self.goal_pos)
        old_path_length = self._shortest_path_length_between(old_pos, self.goal_pos)
        next_path_cell = _bfs_next_cell(self)
        desired_dir = None
        old_turn_alignment = 0.0

        if next_path_cell is not None:
            desired_dir = _desired_dir_to_cell(old_pos, next_path_cell)
            old_turn_alignment = float(min(abs(old_dir - desired_dir), 4 - abs(old_dir - desired_dir)))

        collision = False
        success = False
        if action == self.ACTION_FORWARD:
            next_pos = self._get_forward_pos()

            if self._is_valid_pos(next_pos):
                self.agent_pos = next_pos
            else:
                collision = True

        elif action == self.ACTION_TURN_LEFT:
            self.agent_dir = (self.agent_dir - 1) % 4

        elif action == self.ACTION_TURN_RIGHT:
            self.agent_dir = (self.agent_dir + 1) % 4

        self.step_count += 1

        moved = self.agent_pos != old_pos

        if moved:
            self.steps_since_move = 0
        else:
            self.steps_since_move += 1

        # Markov reward: depends only on current transition/action event.
        # Do not use hidden counters such as steps_since_move here; otherwise
        # identical rendered observations can map to different rewards/dones.
        reward = -self.step_penalty

        if action in [self.ACTION_TURN_LEFT, self.ACTION_TURN_RIGHT]:
            reward -= self.turn_penalty

        # 碰撞惩罚
        if collision:
            reward -= self.collision_penalty

        if self._is_success():
            success = True
            reward += self.goal_reward

        done = False

        if success:
            done = True

        if collision and self.terminate_on_collision:
            done = True

        if self.step_count >= self.max_steps:
            done = True

        obs = self.render()
        new_dist = self._distance(self.agent_pos, self.goal_pos)
        distance_delta = old_dist - new_dist
        new_path_length = self._shortest_path_length_between(self.agent_pos, self.goal_pos)

        if old_path_length is None or new_path_length is None:
            path_length_delta = 0.0
        else:
            path_length_delta = float(old_path_length - new_path_length)

        turn_alignment_delta = 0.0
        if action in [self.ACTION_TURN_LEFT, self.ACTION_TURN_RIGHT] and desired_dir is not None:
            new_turn_alignment = float(
                min(abs(self.agent_dir - desired_dir), 4 - abs(self.agent_dir - desired_dir))
            )
            turn_alignment_delta = old_turn_alignment - new_turn_alignment
            reward += self.turn_shaping_scale * turn_alignment_delta

        reward += self.progress_reward_scale * path_length_delta

        info = {
            "agent_pos": self.agent_pos,
            "agent_dir": self.agent_dir,
            "goal_pos": self.goal_pos,
            "old_pos": old_pos,
            "distance_to_goal": new_dist,
            "distance_delta": distance_delta,
            "path_length_to_goal": new_path_length,
            "path_length_delta": path_length_delta,
            "desired_dir": desired_dir,
            "turn_alignment_delta": turn_alignment_delta,
            "collision": collision,
            "success": success,
            "steps_since_move": self.steps_since_move,
            "step_count": self.step_count,
        }

        return obs, float(reward), done, info

    def render(self):
        return self.renderer.render(
            agent_pos=self.agent_pos,
            agent_dir=self.agent_dir,
            goal_pos=self.goal_pos,
            obstacles=self.obstacles,
        )

    def _get_forward_pos(self) -> Tuple[int, int]:
        dx, dy = self.DIR_TO_DELTA[self.agent_dir]
        x, y = self.agent_pos
        return x + dx, y + dy

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        x, y = pos

        if x < 0 or x >= self.map_width:
            return False

        if y < 0 or y >= self.map_height:
            return False

        if pos in self.obstacles:
            return False

        return True

    def _is_success(self) -> bool:
        return self._distance(self.agent_pos, self.goal_pos) <= self.goal_radius

    def _goal_facing_dir(
        self,
        agent_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
    ) -> int:
        ax, ay = agent_pos
        gx, gy = goal_pos

        dx = gx - ax
        dy = gy - ay

        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 3

        return 2 if dy > 0 else 0

    def _heading_error(
        self,
        agent_pos: Tuple[int, int],
        agent_dir: int,
        goal_pos: Tuple[int, int],
    ) -> float:
        if agent_pos == goal_pos:
            return 0.0

        desired_dir = self._goal_facing_dir(agent_pos, goal_pos)
        diff = abs(agent_dir - desired_dir)

        return float(min(diff, 4 - diff))

    def _distance(
        self,
        pos_a: Tuple[int, int],
        pos_b: Tuple[int, int],
    ) -> float:
        ax, ay = pos_a
        bx, by = pos_b

        return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))

    def _sample_empty_cell(
        self,
        exclude: Set[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        if exclude is None:
            exclude = set()

        while True:
            x = self.rng.randint(0, self.map_width - 1)
            y = self.rng.randint(0, self.map_height - 1)
            pos = (x, y)

            if pos in self.obstacles:
                continue

            if pos in exclude:
                continue

            return pos

    def set_reset_path_length_range(
        self,
        min_path_length: Optional[int] = None,
        max_path_length: Optional[int] = None,
    ):
        if min_path_length is not None and min_path_length < 1:
            raise ValueError("min_path_length should be >= 1.")

        if max_path_length is not None and max_path_length < 1:
            raise ValueError("max_path_length should be >= 1.")

        if (
            min_path_length is not None
            and max_path_length is not None
            and min_path_length > max_path_length
        ):
            raise ValueError("min_path_length should be <= max_path_length.")

        self.reset_min_path_length = min_path_length
        self.reset_max_path_length = max_path_length

    def clear_reset_path_length_range(self):
        self.reset_min_path_length = None
        self.reset_max_path_length = None

    def get_current_shortest_path_length(self) -> Optional[int]:
        if self.agent_pos is None or self.goal_pos is None:
            return None

        return self._shortest_path_length_between(self.agent_pos, self.goal_pos)

    def _sample_agent_goal_pair_with_path_constraints(self):
        for _ in range(self.reset_sample_attempts):
            agent_pos = self._sample_empty_cell()
            goal_pos = self._sample_empty_cell(exclude={agent_pos})

            path_length = self._shortest_path_length_between(agent_pos, goal_pos)
            if path_length is None:
                continue

            if (
                self.reset_min_path_length is not None
                and path_length < self.reset_min_path_length
            ):
                continue

            if (
                self.reset_max_path_length is not None
                and path_length > self.reset_max_path_length
            ):
                continue

            return agent_pos, goal_pos

        agent_pos = self._sample_empty_cell()
        goal_pos = self._sample_empty_cell(exclude={agent_pos})
        return agent_pos, goal_pos

    def _shortest_path_length_between(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[int]:
        if start == goal:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current, dist = queue.popleft()

            if current == goal:
                return dist

            for dx, dy in self.DIR_TO_DELTA.values():
                nxt = (current[0] + dx, current[1] + dy)

                if nxt in visited:
                    continue

                if not self._is_valid_pos(nxt):
                    continue

                visited.add(nxt)
                queue.append((nxt, dist + 1))

        return None


def _bfs_next_cell(env: GridNavEnv) -> Optional[Tuple[int, int]]:
    """
    使用 BFS 在网格上寻找从 agent 到 goal 的最短路径。
    返回路径中的下一个 cell。

    该函数只用于专家策略采样，不作为模型输入。
    """
    start = env.agent_pos
    goal = env.goal_pos

    if start == goal:
        return None

    queue = deque([start])
    parent = {start: None}

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for dx, dy in GridNavEnv.DIR_TO_DELTA.values():
            nx = current[0] + dx
            ny = current[1] + dy
            nxt = (nx, ny)

            if nxt in parent:
                continue

            if not env._is_valid_pos(nxt):
                continue

            parent[nxt] = current
            queue.append(nxt)

    if goal not in parent:
        return None

    path = []
    cur = goal

    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    path.reverse()

    if len(path) < 2:
        return None

    return path[1]


def _desired_dir_to_cell(
    current_pos: Tuple[int, int],
    next_cell: Tuple[int, int],
) -> int:
    dx = next_cell[0] - current_pos[0]
    dy = next_cell[1] - current_pos[1]

    if dx == 0 and dy == -1:
        return 0
    if dx == 1 and dy == 0:
        return 1
    if dx == 0 and dy == 1:
        return 2
    if dx == -1 and dy == 0:
        return 3

    raise ValueError(f"Invalid adjacent cell: {current_pos} -> {next_cell}")


from collections import deque


def _bfs_shortest_path(env: GridNavEnv):
    """
    在当前地图上用 BFS 规划从 agent_pos 到 goal_pos 的最短路径。

    返回：
        path: list[tuple[int, int]]
            包含 start 和 goal。
            如果无路可走，返回 None。
    """
    start = env.agent_pos
    goal = env.goal_pos

    if start == goal:
        return [start]

    queue = deque([start])
    parent = {start: None}

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for direction in [0, 1, 2, 3]:
            dx, dy = GridNavEnv.DIR_TO_DELTA[direction]
            nxt = (current[0] + dx, current[1] + dy)

            if nxt in parent:
                continue

            if not env._is_valid_pos(nxt):
                continue

            parent[nxt] = current
            queue.append(nxt)

    if goal not in parent:
        return None

    path = []
    cur = goal

    while cur is not None:
        path.append(cur)
        cur = parent[cur]

    path.reverse()

    return path


def _direction_to_next_cell(current_pos, next_pos) -> int:
    """
    根据相邻两个 cell 计算应该面向的方向。
    """
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]

    if dx == 0 and dy == -1:
        return 0
    if dx == 1 and dy == 0:
        return 1
    if dx == 0 and dy == 1:
        return 2
    if dx == -1 and dy == 0:
        return 3

    raise ValueError(f"Non-adjacent cells: {current_pos} -> {next_pos}")


def heuristic_action(env: GridNavEnv) -> int:
    """
    BFS 专家动作。

    这个函数只用于采集专家 / DAgger 标签。
    它可以访问 env.agent_pos、env.goal_pos、env.obstacles 等真值状态。
    模型训练时不能把这些真值状态作为输入。
    """
    path = _bfs_shortest_path(env)

    if path is None or len(path) <= 1:
        return GridNavEnv.ACTION_TURN_LEFT

    current_pos = path[0]
    next_pos = path[1]

    desired_dir = _direction_to_next_cell(current_pos, next_pos)

    if env.agent_dir == desired_dir:
        return GridNavEnv.ACTION_FORWARD

    left_dir = (env.agent_dir - 1) % 4
    right_dir = (env.agent_dir + 1) % 4

    if left_dir == desired_dir:
        return GridNavEnv.ACTION_TURN_LEFT

    if right_dir == desired_dir:
        return GridNavEnv.ACTION_TURN_RIGHT

    # desired_dir 在正后方时，固定右转，避免随机标签
    return GridNavEnv.ACTION_TURN_RIGHT


def collect_episode_heuristic(env: GridNavEnv, max_steps: int):
    """
    使用 BFS 专家策略采集 episode。
    """
    obs_list = []
    action_list = []
    reward_list = []
    done_list = []
    success_list = []
    collision_list = []
    obs = env.reset()

    obs_list.append(obs)
    action_list.append(GridNavEnv.ACTION_FORWARD)
    reward_list.append(0.0)
    done_list.append(False)
    success_list.append(False)
    collision_list.append(False)

    total_reward = 0.0
    success = False
    collision = False

    for _ in range(max_steps):
        action = int(heuristic_action(env))

        next_obs, reward, done, info = env.step(action)

        obs_list.append(next_obs)
        action_list.append(action)
        reward_list.append(float(reward))
        done_list.append(bool(done))

        total_reward += float(reward)
        success = bool(info.get("success", False))
        collision = bool(info.get("collision", False))
        success_list.append(success)
        collision_list.append(collision)

        if done:
            break

        obs = next_obs

    return {
        "obs": np.stack(obs_list, axis=0),
        "actions": np.array(action_list, dtype=np.int64),
        "rewards": np.array(reward_list, dtype=np.float32),
        "dones": np.array(done_list, dtype=np.bool_),
        "total_reward": total_reward,
        "success": success,
        "collision": collision,
        "length": len(obs_list),
        "successes": np.array(success_list, dtype=np.bool_),
        "collisions": np.array(collision_list, dtype=np.bool_),
    }
