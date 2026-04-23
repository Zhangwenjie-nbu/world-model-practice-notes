import os
import random
import yaml
import numpy as np
import torch

from datasets.moving_mnist import build_dataloader
from models.world_model import WorldModel
from training.trainer import train_one_epoch
from training.eval import evaluate_one_epoch
from utils.checkpoint import ensure_dir, save_checkpoint
from utils.logger import SimpleLogger


def set_seed(seed: int):
    """
    固定随机种子，保证实验可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str):
    """
    读取 yaml 配置文件。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: dict, device: torch.device) -> WorldModel:
    """
    根据配置构建 WorldModel。
    """
    model_cfg = config["model"]

    model = WorldModel(
        image_channels=model_cfg["image_channels"],
        image_size=model_cfg["image_size"],
        embedding_dim=model_cfg["embedding_dim"],
        deter_dim=model_cfg["deter_dim"],
        stoch_dim=model_cfg["stoch_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        min_std=model_cfg["min_std"],
    )

    model = model.to(device)
    return model


def main():
    config = load_config("configs/moving_mnist.yaml")

    set_seed(config["seed"])

    device_str = config["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        print("警告：配置为 cuda，但当前环境不可用，自动切换到 cpu。")
        device_str = "cpu"

    device = torch.device(device_str)

    logger = SimpleLogger()
    logger.log(f"使用设备: {device}")

    dataset_cfg = config["dataset"]
    train_cfg = config["train"]
    eval_cfg = config["eval"]
    loss_cfg = config["loss"]
    ckpt_cfg = config["checkpoint"]

    # 构建训练集与验证集
    train_dataset, train_loader = build_dataloader(
        data_path=dataset_cfg["train_path"],
        seq_len=dataset_cfg["seq_len"],
        context_len=dataset_cfg["context_len"],
        pred_len=dataset_cfg["pred_len"],
        image_size=dataset_cfg["image_size"],
        batch_size=train_cfg["batch_size"],
        shuffle=train_cfg["shuffle"],
        drop_last=train_cfg["drop_last"],
        num_workers=dataset_cfg["num_workers"],
    )

    test_dataset, test_loader = build_dataloader(
        data_path=dataset_cfg["test_path"],
        seq_len=dataset_cfg["seq_len"],
        context_len=dataset_cfg["context_len"],
        pred_len=dataset_cfg["pred_len"],
        image_size=dataset_cfg["image_size"],
        batch_size=eval_cfg["batch_size"],
        shuffle=eval_cfg["shuffle"],
        drop_last=eval_cfg["drop_last"],
        num_workers=dataset_cfg["num_workers"],
    )

    logger.log(f"训练集大小: {len(train_dataset)}")
    logger.log(f"验证集大小: {len(test_dataset)}")

    # 构建模型
    model = build_model(config, device=device)
    logger.log("WorldModel 构建完成。")

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
    )
    logger.log("优化器构建完成。")

    # checkpoint 目录
    ensure_dir(ckpt_cfg["save_dir"])

    num_epochs = train_cfg["num_epochs"]
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        logger.log(f"开始训练 Epoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            recon_loss_type=loss_cfg["recon_loss_type"],
            kl_weight=loss_cfg["kl_weight"],
            free_nats=loss_cfg["free_nats"],
            future_loss_weight=loss_cfg["future_loss_weight"],
            grad_clip_norm=train_cfg["grad_clip_norm"],
            log_interval=train_cfg["log_interval"],
            epoch=epoch,
        )

        global_step += len(train_loader)

        eval_metrics = evaluate_one_epoch(
            model=model,
            loader=test_loader,
            device=device,
            recon_loss_type=loss_cfg["recon_loss_type"],
            kl_weight=loss_cfg["kl_weight"],
            free_nats=loss_cfg["free_nats"],
            future_loss_weight=loss_cfg["future_loss_weight"],
            epoch=epoch,
        )

        logger.log(
            f"Epoch {epoch} 完成 | "
            f"Train total={train_metrics['total_loss']:.6f}, "
            f"Train future={train_metrics['future_loss']:.6f}, "
            f"Eval total={eval_metrics['total_loss']:.6f}, "
            f"Eval future={eval_metrics['future_loss']:.6f}, "
            f"Eval future_mse={eval_metrics['future_mse']:.6f}"
        )

        # 保存 checkpoint
        if epoch % ckpt_cfg["save_every"] == 0:
            save_path = os.path.join(
                ckpt_cfg["save_dir"],
                f"epoch_{epoch:03d}.pt",
            )

            save_checkpoint(
                save_path=save_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                extra={
                    "train_metrics": train_metrics,
                    "eval_metrics": eval_metrics,
                    "config": config,
                },
            )
            logger.log(f"Checkpoint 已保存到: {save_path}")

    logger.log("训练结束。")


if __name__ == "__main__":
    main()