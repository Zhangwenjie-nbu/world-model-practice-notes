# scripts/test_env.py

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# 让脚本可以从项目根目录导入 envs 模块
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from envs.grid_nav_env import GridNavEnv


def save_png_image (image: np.ndarray, save_path: str):
    """
    保存 PNG 格式图片。

    参数：
        image:
            uint8 RGB 图像，形状为 H x W x 3。

        save_path:
            保存路径，例如 outputs/test_obs.png。
    """
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8

    img = Image.fromarray(image, mode="RGB")
    img.save(save_path)


def main():
    env = GridNavEnv(
        map_width=10,
        map_height=10,
        image_size=64,
        max_steps=50,
        random_reset=True,
        seed=42,
    )

    obs = env.reset()

    print("=" * 60)
    print("环境 reset 完成")
    print(f"obs shape: {obs.shape}")
    print(f"obs dtype: {obs.dtype}")
    print(f"agent_pos: {env.agent_pos}")
    print(f"agent_dir: {env.agent_dir}")
    print(f"goal_pos: {env.goal_pos}")
    print("=" * 60)

    output_dir = ROOT_DIR / "outputs"
    os.makedirs(output_dir, exist_ok=True)

    save_path = output_dir / "test_obs.png"
    save_png_image(obs, str(save_path))
    print(f"初始观测图像已保存到: {save_path}")

    total_reward = 0.0

    for t in range(20):
        action = np.random.randint(0, env.num_actions)

        obs, reward, done, info = env.step(action)
        total_reward += reward

        print(
            f"step={t:02d}, "
            f"action={action}, "
            f"reward={reward:.3f}, "
            f"done={done}, "
            f"pos={info['agent_pos']}, "
            f"dir={info['agent_dir']}, "
            f"dist={info['distance_to_goal']:.3f}, "
            f"collision={info['collision']}, "
            f"success={info['success']}"
        )

        if done:
            print("episode finished.")
            break

    print("=" * 60)
    print(f"total_reward: {total_reward:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()