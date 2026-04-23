from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class WorldModelLossOutput:
    """
    world model 损失输出结构。

    字段说明：
        total_loss:
            总损失，标量 tensor。

        observe_recon_loss:
            整段 observe reconstruction 损失，标量 tensor。

        kl_loss:
            经过 free_nats 处理后的 KL 损失，标量 tensor。

        raw_kl_loss:
            未经过 free_nats 处理的原始 KL 损失，标量 tensor。

        future_loss:
            future prediction loss，标量 tensor。
            如果当前没有启用 future loss，则为 None。

        loss_dict:
            用于日志打印的 Python 字典。
    """

    total_loss: torch.Tensor
    observe_recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    raw_kl_loss: torch.Tensor
    future_loss: Optional[torch.Tensor]
    loss_dict: Dict[str, float]


def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    计算图像重建损失。

    参数：
        recon:
            模型输出图像，shape 为 [B, T, C, H, W]。

        target:
            真实图像，shape 为 [B, T, C, H, W]。

        loss_type:
            当前支持：
                - "mse"
                - "bce"
    """
    if recon.shape != target.shape:
        raise ValueError(
            f"reconstruction_loss 要求 recon 和 target shape 一致，"
            f"但 recon={recon.shape}, target={target.shape}"
        )

    if loss_type == "mse":
        loss = F.mse_loss(recon, target, reduction="mean")
    elif loss_type == "bce":
        loss = F.binary_cross_entropy(recon, target, reduction="mean")
    else:
        raise ValueError(
            f"不支持的 reconstruction loss 类型: {loss_type}，"
            f"当前仅支持 'mse' 和 'bce'。"
        )

    return loss


def gaussian_kl_divergence(
    post_mean: torch.Tensor,
    post_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    reduce: bool = True,
) -> torch.Tensor:
    """
    计算两个对角高斯分布之间的 KL 散度。

    计算：
        KL(q || p)
    """
    if not (
        post_mean.shape == post_std.shape == prior_mean.shape == prior_std.shape
    ):
        raise ValueError(
            "gaussian_kl_divergence 要求所有输入 shape 一致，"
            f"但得到 post_mean={post_mean.shape}, post_std={post_std.shape}, "
            f"prior_mean={prior_mean.shape}, prior_std={prior_std.shape}"
        )

    eps = 1e-8
    post_std = torch.clamp(post_std, min=eps)
    prior_std = torch.clamp(prior_std, min=eps)

    kl_per_dim = (
        torch.log(prior_std)
        - torch.log(post_std)
        + (post_std.pow(2) + (post_mean - prior_mean).pow(2))
        / (2.0 * prior_std.pow(2))
        - 0.5
    )

    kl = kl_per_dim.sum(dim=-1)

    if reduce:
        kl = kl.mean()

    return kl


def apply_free_nats(
    kl_loss: torch.Tensor,
    free_nats: float,
) -> torch.Tensor:
    """
    对 KL loss 应用 free nats 下界。
    """
    if free_nats <= 0:
        return kl_loss

    free_nats_tensor = torch.tensor(
        free_nats,
        dtype=kl_loss.dtype,
        device=kl_loss.device,
    )

    return torch.clamp(kl_loss, min=free_nats_tensor)


def world_model_loss(
    observe_recon: torch.Tensor,
    observe_target: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    post_mean: torch.Tensor,
    post_std: torch.Tensor,
    recon_loss_type: str = "mse",
    kl_weight: float = 1.0,
    free_nats: float = 1.0,
    future_recon: Optional[torch.Tensor] = None,
    future_target: Optional[torch.Tensor] = None,
    future_loss_weight: float = 0.0,
) -> WorldModelLossOutput:
    """
    计算联合训练的 world model 总损失。

    总损失：
        total =
            observe_recon_loss
            + kl_weight * kl_loss
            + future_loss_weight * future_loss

    其中 future_loss 可选。
    """
    observe_recon_loss = reconstruction_loss(
        recon=observe_recon,
        target=observe_target,
        loss_type=recon_loss_type,
    )

    raw_kl_loss = gaussian_kl_divergence(
        post_mean=post_mean,
        post_std=post_std,
        prior_mean=prior_mean,
        prior_std=prior_std,
        reduce=True,
    )

    kl_loss = apply_free_nats(
        kl_loss=raw_kl_loss,
        free_nats=free_nats,
    )

    future_loss = None
    total_loss = observe_recon_loss + kl_weight * kl_loss

    if future_recon is not None or future_target is not None:
        if future_recon is None or future_target is None:
            raise ValueError("future_recon 和 future_target 必须同时提供，或同时为 None。")

        future_loss = reconstruction_loss(
            recon=future_recon,
            target=future_target,
            loss_type=recon_loss_type,
        )

        total_loss = total_loss + future_loss_weight * future_loss

    loss_dict = {
        "total_loss": float(total_loss.detach().cpu().item()),
        "observe_recon_loss": float(observe_recon_loss.detach().cpu().item()),
        "kl_loss": float(kl_loss.detach().cpu().item()),
        "raw_kl_loss": float(raw_kl_loss.detach().cpu().item()),
        "kl_weight": float(kl_weight),
        "free_nats": float(free_nats),
        "future_loss_weight": float(future_loss_weight),
    }

    if future_loss is not None:
        loss_dict["future_loss"] = float(future_loss.detach().cpu().item())
    else:
        loss_dict["future_loss"] = 0.0

    return WorldModelLossOutput(
        total_loss=total_loss,
        observe_recon_loss=observe_recon_loss,
        kl_loss=kl_loss,
        raw_kl_loss=raw_kl_loss,
        future_loss=future_loss,
        loss_dict=loss_dict,
    )