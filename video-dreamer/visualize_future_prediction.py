import os
import yaml
import torch

from datasets.moving_mnist import build_dataloader
from models.world_model import WorldModel
from utils.visualize import save_future_prediction_comparison


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


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    """
    加载 checkpoint。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


@torch.no_grad()
def main():
    config = load_config("configs/moving_mnist.yaml")

    device_str = config["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        print("警告：配置为 cuda，但当前环境不可用，自动切换到 cpu。")
        device_str = "cpu"
    device = torch.device(device_str)

    dataset_cfg = config["dataset"]
    eval_cfg = config["eval"]
    output_dir = config["output"]["vis_dir"]

    # 改成你自己的 checkpoint 路径
    checkpoint_path = "outputs/checkpoints/epoch_010.pt"

    _, test_loader = build_dataloader(
        data_path=dataset_cfg["test_path"],
        seq_len=dataset_cfg["seq_len"],
        context_len=dataset_cfg["context_len"],
        pred_len=dataset_cfg["pred_len"],
        image_size=dataset_cfg["image_size"],
        batch_size=eval_cfg["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=dataset_cfg["num_workers"],
    )

    model = build_model(config, device=device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()

    batch = next(iter(test_loader))

    context = batch["context"].to(device)  # [B, K, C, H, W]
    target = batch["target"].to(device)    # [B, H, C, H, W]

    pred_output = model.predict_future(
        context_video=context,
        pred_len=target.shape[1],
    )

    pred = pred_output.future_predictions

    os.makedirs(output_dir, exist_ok=True)

    num_samples = min(config["eval"]["num_visualize_samples"], context.shape[0])

    for i in range(num_samples):
        save_path = os.path.join(output_dir, f"future_prediction_{i}.png")

        save_future_prediction_comparison(
            context_video=context[i],
            target_video=target[i],
            pred_video=pred[i],
            save_path=save_path,
        )

        print(f"样本 {i} 的未来预测对比图已保存到: {save_path}")


if __name__ == "__main__":
    main()