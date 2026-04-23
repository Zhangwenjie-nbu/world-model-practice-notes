import torch

from models.world_model import WorldModel


def main():
    """
    测试 WorldModel 是否能对整段视频执行完整前向。
    """
    batch_size = 8
    seq_len = 20
    channels = 1
    image_size = 64

    model = WorldModel(
        image_channels=channels,
        image_size=image_size,
        embedding_dim=128,
        deter_dim=128,
        stoch_dim=32,
        hidden_dim=256,
        min_std=0.1,
    )

    video = torch.rand(batch_size, seq_len, channels, image_size, image_size)

    output = model.forward_observe(video)

    print("输入视频 shape:", video.shape)
    print("embedding shape:", output.embeddings.shape)
    print("rollout features shape:", output.rollout.features.shape)
    print("reconstruction shape:", output.reconstructions.shape)

    assert output.embeddings.shape == (batch_size, seq_len, 128)
    assert output.rollout.features.shape == (batch_size, seq_len, 160)
    assert output.reconstructions.shape == (batch_size, seq_len, channels, image_size, image_size)

    print("WorldModel 前向测试通过。")


if __name__ == "__main__":
    main()