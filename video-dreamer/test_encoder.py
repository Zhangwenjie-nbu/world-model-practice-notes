import torch

from models.encoder import ImageEncoder, encode_video_frames


def test_single_frame_encoder():
    """
    测试 Encoder 是否能正确处理单帧图像 batch。
    """
    batch_size = 16
    channels = 1
    image_size = 64
    embedding_dim = 128

    encoder = ImageEncoder(
        in_channels=channels,
        embedding_dim=embedding_dim,
        image_size=image_size,
    )

    x = torch.randn(batch_size, channels, image_size, image_size)
    embedding = encoder(x)

    print("单帧输入 shape:", x.shape)
    print("单帧 embedding shape:", embedding.shape)

    assert embedding.shape == (batch_size, embedding_dim), (
        f"Encoder 输出 shape 错误，期望 {(batch_size, embedding_dim)}，"
        f"实际得到 {embedding.shape}"
    )


def test_video_encoder():
    """
    测试 Encoder 是否能正确处理视频 batch。
    """
    batch_size = 16
    seq_len = 20
    channels = 1
    image_size = 64
    embedding_dim = 128

    encoder = ImageEncoder(
        in_channels=channels,
        embedding_dim=embedding_dim,
        image_size=image_size,
    )

    video = torch.randn(batch_size, seq_len, channels, image_size, image_size)
    embeddings = encode_video_frames(encoder, video)

    print("视频输入 shape:", video.shape)
    print("视频 embedding shape:", embeddings.shape)

    assert embeddings.shape == (batch_size, seq_len, embedding_dim), (
        f"视频 Encoder 输出 shape 错误，期望 {(batch_size, seq_len, embedding_dim)}，"
        f"实际得到 {embeddings.shape}"
    )


def main():
    test_single_frame_encoder()
    test_video_encoder()
    print("Encoder 测试通过。")


if __name__ == "__main__":
    main()