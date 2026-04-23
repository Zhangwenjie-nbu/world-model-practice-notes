import yaml
import torch

from datasets.moving_mnist import build_dataloader
from models.encoder import ImageEncoder, encode_video_frames


def load_config(config_path: str):
    """
    读取 yaml 配置文件。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main():
    config = load_config("configs/moving_mnist.yaml")

    dataset_cfg = config["dataset"]
    train_cfg = config["train"]
    model_cfg = config["model"]

    _, train_loader = build_dataloader(
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

    encoder = ImageEncoder(
        in_channels=model_cfg["image_channels"],
        embedding_dim=model_cfg["embedding_dim"],
        image_size=model_cfg["image_size"],
    )

    batch = next(iter(train_loader))
    video = batch["video"]

    embeddings = encode_video_frames(encoder, video)

    print("真实视频 batch shape:", video.shape)
    print("真实视频 embedding shape:", embeddings.shape)

    expected_shape = (
        train_cfg["batch_size"],
        dataset_cfg["seq_len"],
        model_cfg["embedding_dim"],
    )

    assert embeddings.shape == expected_shape, (
        f"真实数据 Encoder 输出 shape 错误，期望 {expected_shape}，"
        f"实际得到 {embeddings.shape}"
    )

    print("真实数据 Encoder 测试通过。")


if __name__ == "__main__":
    main()