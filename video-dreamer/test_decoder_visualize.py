import os
import torch

from models.decoder import ImageDecoder, decode_video_features
from utils.visualize import save_video_grid


def main():
    """
    保存未训练 Decoder 的随机输出图像。

    注意：
        因为 Decoder 还没有训练，所以图像没有实际语义。
        这个脚本只用于检查输出范围和可视化流程是否正常。
    """
    batch_size = 1
    seq_len = 20
    feature_dim = 160
    out_channels = 1
    image_size = 64

    decoder = ImageDecoder(
        feature_dim=feature_dim,
        out_channels=out_channels,
        image_size=image_size,
    )

    features = torch.randn(batch_size, seq_len, feature_dim)
    recons = decode_video_features(decoder, features)

    save_dir = "outputs/vis"
    os.makedirs(save_dir, exist_ok=True)

    save_video_grid(
        video=recons[0],
        save_path=os.path.join(save_dir, "random_decoder_output.png"),
        title="Random decoder output",
    )

    print(f"随机 Decoder 输出已保存到: {save_dir}/random_decoder_output.png")


if __name__ == "__main__":
    main()