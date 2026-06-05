"""
项目主入口（第一课版本）
当前阶段只做一件事：
1. 检查项目目录和配置是否正确
2. 打印项目的任务定义，确保后续开发目标明确

后续课程中，这里可以扩展成统一入口：
- 训练 world model
- 训练 agent
- 评估
- 可视化
"""

from pathlib import Path


def print_project_intro():
    """打印当前项目的基本说明"""
    print("=" * 60)
    print("项目名称：视觉导航 / 避障 World Model Agent")
    print("任务目标：根据视觉观测进行导航，避开障碍并到达目标")
    print("第一版环境：2D 俯视图、离散动作、RGB 图像输入")
    print("后续模块：Encoder / RSSM / Reward Model / Actor / Critic")
    print("=" * 60)


def check_project_dirs():
    """检查关键目录是否存在，不存在则提示"""
    required_dirs = [
        "configs",
        "envs",
        "models",
        "trainers",
        "memory",
        "scripts",
        "tools",
        "outputs",
    ]

    print("\n[检查项目目录]")
    root = Path(".")
    for d in required_dirs:
        path = root / d
        if path.exists():
            print(f"[OK] 目录存在: {d}")
        else:
            print(f"[WARN] 目录不存在: {d}")


if __name__ == "__main__":
    print_project_intro()
    check_project_dirs()