# envs/renderer.py

import numpy as np


class GridRenderer:
    """
    2D 网格环境渲染器。

    功能：
    1. 将障碍物、目标点、智能体位置渲染为 RGB 图像；
    2. 输出形状为 image_size x image_size x 3 的 uint8 图像；
    3. 不依赖 cv2 / PIL，降低环境依赖。
    """

    def __init__(self, map_width: int, map_height: int, image_size: int = 64):
        self.map_width = map_width
        self.map_height = map_height
        self.image_size = image_size

        # 每个网格单元在图像中的像素大小
        self.cell_w = image_size // map_width
        self.cell_h = image_size // map_height

    def render(self, agent_pos, agent_dir, goal_pos, obstacles):
        """
        渲染当前环境状态。

        参数：
            agent_pos: tuple[int, int]
                智能体位置，例如 (x, y)

            agent_dir: int
                智能体朝向：
                0 = up
                1 = right
                2 = down
                3 = left

            goal_pos: tuple[int, int]
                目标位置

            obstacles: set[tuple[int, int]]
                障碍物位置集合

        返回：
            image: np.ndarray
                RGB 图像，形状为 (image_size, image_size, 3)
        """

        # 白色背景
        image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255

        # 绘制网格线，方便调试观察
        self._draw_grid(image)

        # 绘制障碍物：黑色
        for obs in obstacles:
            self._draw_cell(image, obs, color=(0, 0, 0))

        # 绘制目标：红色
        self._draw_cell(image, goal_pos, color=(255, 0, 0))

        # 绘制 agent：蓝色
        self._draw_cell(image, agent_pos, color=(0, 80, 255))

        # 绘制 agent 朝向：浅蓝色小块
        self._draw_agent_direction(image, agent_pos, agent_dir)

        return image

    def _draw_cell(self, image, pos, color):
        """
        将某个网格单元绘制成指定颜色。
        """
        x, y = pos

        x0 = x * self.cell_w
        y0 = y * self.cell_h
        x1 = min((x + 1) * self.cell_w, self.image_size)
        y1 = min((y + 1) * self.cell_h, self.image_size)

        image[y0:y1, x0:x1, :] = np.array(color, dtype=np.uint8)

    def _draw_grid(self, image):
        """
        绘制浅灰色网格线，便于可视化。
        """
        grid_color = np.array([220, 220, 220], dtype=np.uint8)

        for x in range(self.map_width + 1):
            px = min(x * self.cell_w, self.image_size - 1)
            image[:, px:px + 1, :] = grid_color

        for y in range(self.map_height + 1):
            py = min(y * self.cell_h, self.image_size - 1)
            image[py:py + 1, :, :] = grid_color

    def _draw_agent_direction(self, image, agent_pos, agent_dir):
        """
        在 agent 所在格子内部绘制一个小色块，用于表示朝向。
        """
        x, y = agent_pos

        cx = x * self.cell_w + self.cell_w // 2
        cy = y * self.cell_h + self.cell_h // 2

        offset = max(1, min(self.cell_w, self.cell_h) // 4)

        if agent_dir == 0:      # up
            dx, dy = 0, -offset
        elif agent_dir == 1:    # right
            dx, dy = offset, 0
        elif agent_dir == 2:    # down
            dx, dy = 0, offset
        elif agent_dir == 3:    # left
            dx, dy = -offset, 0
        else:
            raise ValueError(f"Invalid agent_dir: {agent_dir}")

        px = int(np.clip(cx + dx, 0, self.image_size - 1))
        py = int(np.clip(cy + dy, 0, self.image_size - 1))

        # 朝向点使用浅蓝色
        r = max(1, offset // 2)
        image[max(0, py - r):min(self.image_size, py + r + 1),
              max(0, px - r):min(self.image_size, px + r + 1),
              :] = np.array([120, 200, 255], dtype=np.uint8)