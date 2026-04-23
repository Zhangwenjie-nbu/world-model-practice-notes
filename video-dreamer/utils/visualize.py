import os
import matplotlib.pyplot as plt
import torch


def save_future_prediction_comparison(
    context_video: torch.Tensor,
    target_video: torch.Tensor,
    pred_video: torch.Tensor,
    save_path: str,
):
    """
    保存未来帧预测对比图。

    图像布局：
        第 1 行：context 真值
        第 2 行：future 真值
        第 3 行：future 预测

    参数：
        context_video:
            前 K 帧真实视频，shape 为 [K, 1, H, W] 或 [K, H, W]

        target_video:
            后 H 帧真实视频，shape 为 [H, 1, H, W] 或 [H, H, W]

        pred_video:
            后 H 帧预测视频，shape 为 [H, 1, H, W] 或 [H, H, W]

        save_path:
            保存路径。
    """
    if isinstance(context_video, torch.Tensor):
        context_video = context_video.detach().cpu()
    if isinstance(target_video, torch.Tensor):
        target_video = target_video.detach().cpu()
    if isinstance(pred_video, torch.Tensor):
        pred_video = pred_video.detach().cpu()

    if context_video.ndim == 4:
        context_video = context_video.squeeze(1)
    if target_video.ndim == 4:
        target_video = target_video.squeeze(1)
    if pred_video.ndim == 4:
        pred_video = pred_video.squeeze(1)

    if context_video.ndim != 3 or target_video.ndim != 3 or pred_video.ndim != 3:
        raise ValueError(
            f"输入维度错误: context={context_video.shape}, "
            f"target={target_video.shape}, pred={pred_video.shape}"
        )

    K = context_video.shape[0]
    H = target_video.shape[0]
    H_pred = pred_video.shape[0]

    if H != H_pred:
        raise ValueError(
            f"target 和 pred 的时间长度必须一致，但得到 target={H}, pred={H_pred}"
        )

    cols = max(K, H)

    fig, axes = plt.subplots(3, cols, figsize=(cols * 2, 6))

    if cols == 1:
        axes = [[axes[0]], [axes[1]], [axes[2]]]

    # 第 1 行：context
    for i in range(cols):
        ax = axes[0][i]
        if i < K:
            ax.imshow(context_video[i], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"CTX {i+1}")
        ax.axis("off")

    # 第 2 行：future ground truth
    for i in range(cols):
        ax = axes[1][i]
        if i < H:
            ax.imshow(target_video[i], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"GT {i+1}")
        ax.axis("off")

    # 第 3 行：future prediction
    for i in range(cols):
        ax = axes[2][i]
        if i < H_pred:
            ax.imshow(pred_video[i], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"PRED {i+1}")
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
