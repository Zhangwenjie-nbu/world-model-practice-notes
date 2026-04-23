import os
import torch


def ensure_dir(path: str):
    """
    如果目录不存在，则自动创建。
    """
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
    save_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    extra: dict = None,
):
    """
    保存训练 checkpoint。

    参数：
        save_path:
            checkpoint 文件路径。

        model:
            待保存模型。

        optimizer:
            待保存优化器。

        epoch:
            当前 epoch 编号。

        global_step:
            当前全局 step。

        extra:
            额外想保存的信息。
    """
    ensure_dir(os.path.dirname(save_path))

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    map_location: str = "cpu",
):
    """
    加载 checkpoint。

    返回：
        checkpoint 字典。
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint 文件不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint