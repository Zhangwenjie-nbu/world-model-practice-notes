from typing import Dict

import torch
from torch.nn.utils import clip_grad_norm_

from training.losses import world_model_loss
from utils.logger import AverageMeter


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    将 batch 中的所有 tensor 移动到指定设备。
    """
    output = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            output[k] = v.to(device, non_blocking=True)
        else:
            output[k] = v
    return output


def train_one_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    recon_loss_type: str = "mse",
    kl_weight: float = 1.0,
    free_nats: float = 1.0,
    future_loss_weight: float = 0.0,
    grad_clip_norm: float = 100.0,
) -> Dict[str, float]:
    """
    执行一步联合训练。

    训练目标包含：
        1. observe reconstruction loss
        2. KL loss
        3. future prediction loss
    """
    model.train()

    batch = move_batch_to_device(batch, device)

    video = batch["video"]      # [B, T, C, H, W]
    context = batch["context"]  # [B, K, C, H, W]
    target = batch["target"]    # [B, H, C, H, W]

    optimizer.zero_grad()

    # =========================
    # 分支 A：整段 observe reconstruction
    # =========================
    observe_output = model.forward_observe(video)

    # =========================
    # 分支 B：context -> future prediction
    # =========================
    future_output = model.predict_future(
        context_video=context,
        pred_len=target.shape[1],
    )

    loss_output = world_model_loss(
        observe_recon=observe_output.reconstructions,
        observe_target=video,
        prior_mean=observe_output.rollout.prior_means,
        prior_std=observe_output.rollout.prior_stds,
        post_mean=observe_output.rollout.post_means,
        post_std=observe_output.rollout.post_stds,
        recon_loss_type=recon_loss_type,
        kl_weight=kl_weight,
        free_nats=free_nats,
        future_recon=future_output.future_predictions,
        future_target=target,
        future_loss_weight=future_loss_weight,
    )

    loss_output.total_loss.backward()

    if grad_clip_norm is not None and grad_clip_norm > 0:
        clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

    optimizer.step()

    return loss_output.loss_dict


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    recon_loss_type: str = "mse",
    kl_weight: float = 1.0,
    free_nats: float = 1.0,
    future_loss_weight: float = 0.0,
    grad_clip_norm: float = 100.0,
    log_interval: int = 50,
    epoch: int = 0,
) -> Dict[str, float]:
    """
    执行一个 epoch 的联合训练。
    """
    total_meter = AverageMeter()
    observe_recon_meter = AverageMeter()
    kl_meter = AverageMeter()
    raw_kl_meter = AverageMeter()
    future_meter = AverageMeter()

    for step, batch in enumerate(loader, start=1):
        loss_dict = train_one_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            recon_loss_type=recon_loss_type,
            kl_weight=kl_weight,
            free_nats=free_nats,
            future_loss_weight=future_loss_weight,
            grad_clip_norm=grad_clip_norm,
        )

        total_meter.update(loss_dict["total_loss"])
        observe_recon_meter.update(loss_dict["observe_recon_loss"])
        kl_meter.update(loss_dict["kl_loss"])
        raw_kl_meter.update(loss_dict["raw_kl_loss"])
        future_meter.update(loss_dict["future_loss"])

        if step % log_interval == 0 or step == 1 or step == len(loader):
            print(
                f"[Train][Epoch {epoch}] "
                f"Step {step}/{len(loader)} | "
                f"total={total_meter.avg:.6f}, "
                f"obs_recon={observe_recon_meter.avg:.6f}, "
                f"future={future_meter.avg:.6f}, "
                f"kl={kl_meter.avg:.6f}, "
                f"raw_kl={raw_kl_meter.avg:.6f}"
            )

    epoch_loss_dict = {
        "total_loss": total_meter.avg,
        "observe_recon_loss": observe_recon_meter.avg,
        "future_loss": future_meter.avg,
        "kl_loss": kl_meter.avg,
        "raw_kl_loss": raw_kl_meter.avg,
    }

    return epoch_loss_dict