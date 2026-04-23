from typing import Dict

import torch
import torch.nn.functional as F

from training.losses import world_model_loss
from training.trainer import move_batch_to_device
from utils.logger import AverageMeter


@torch.no_grad()
def evaluate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    recon_loss_type: str = "mse",
    kl_weight: float = 1.0,
    free_nats: float = 1.0,
    future_loss_weight: float = 0.0,
    epoch: int = 0,
) -> Dict[str, float]:
    """
    执行一个 epoch 的联合验证。
    """
    model.eval()

    total_meter = AverageMeter()
    observe_recon_meter = AverageMeter()
    kl_meter = AverageMeter()
    raw_kl_meter = AverageMeter()
    future_meter = AverageMeter()
    future_mse_meter = AverageMeter()

    for _, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)

        video = batch["video"]
        context = batch["context"]
        target = batch["target"]

        observe_output = model.forward_observe(video)

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

        future_mse = F.mse_loss(
            future_output.future_predictions,
            target,
            reduction="mean",
        )

        total_meter.update(loss_output.loss_dict["total_loss"])
        observe_recon_meter.update(loss_output.loss_dict["observe_recon_loss"])
        future_meter.update(loss_output.loss_dict["future_loss"])
        kl_meter.update(loss_output.loss_dict["kl_loss"])
        raw_kl_meter.update(loss_output.loss_dict["raw_kl_loss"])
        future_mse_meter.update(float(future_mse.detach().cpu().item()))

    epoch_loss_dict = {
        "total_loss": total_meter.avg,
        "observe_recon_loss": observe_recon_meter.avg,
        "future_loss": future_meter.avg,
        "future_mse": future_mse_meter.avg,
        "kl_loss": kl_meter.avg,
        "raw_kl_loss": raw_kl_meter.avg,
    }

    print(
        f"[Eval ][Epoch {epoch}] "
        f"total={epoch_loss_dict['total_loss']:.6f}, "
        f"obs_recon={epoch_loss_dict['observe_recon_loss']:.6f}, "
        f"future={epoch_loss_dict['future_loss']:.6f}, "
        f"future_mse={epoch_loss_dict['future_mse']:.6f}, "
        f"kl={epoch_loss_dict['kl_loss']:.6f}, "
        f"raw_kl={epoch_loss_dict['raw_kl_loss']:.6f}"
    )

    return epoch_loss_dict