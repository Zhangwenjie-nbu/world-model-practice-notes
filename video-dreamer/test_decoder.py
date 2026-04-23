import torch

from models.decoder import ImageDecoder, decode_video_features


def test_single_feature_decoder():
    """
    测试 Decoder 是否能正确处理单个时间步的 latent feature batch。
    """
    batch_size = 16
    feature_dim = 160
    out_channels = 1
    image_size = 64

    decoder = ImageDecoder(
        feature_dim=feature_dim,
        out_channels=out_channels,
        image_size=image_size,
    )

    feature = torch.randn(batch_size, feature_dim)
    recon = decoder(feature)

    print("单步 feature shape:", feature.shape)
    print("单步重建图像 shape:", recon.shape)
    print("重建图像最小值:", recon.min().item())
    print("重建图像最大值:", recon.max().item())

    expected_shape = (batch_size, out_channels, image_size, image_size)

    assert recon.shape == expected_shape, (
        f"Decoder 输出 shape 错误，期望 {expected_shape}，实际得到 {recon.shape}"
    )

    assert recon.min().item() >= 0.0 and recon.max().item() <= 1.0, (
        "Decoder 输出不在 [0, 1] 范围内，请检查最后是否使用 Sigmoid。"
    )


def test_video_feature_decoder():
    """
    测试 Decoder 是否能正确处理一整段 latent feature 序列。
    """
    batch_size = 16
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

    print("视频 feature shape:", features.shape)
    print("视频重建 shape:", recons.shape)
    print("视频重建最小值:", recons.min().item())
    print("视频重建最大值:", recons.max().item())

    expected_shape = (batch_size, seq_len, out_channels, image_size, image_size)

    assert recons.shape == expected_shape, (
        f"视频 Decoder 输出 shape 错误，期望 {expected_shape}，实际得到 {recons.shape}"
    )

    assert recons.min().item() >= 0.0 and recons.max().item() <= 1.0, (
        "视频 Decoder 输出不在 [0, 1] 范围内，请检查最后是否使用 Sigmoid。"
    )


def main():
    test_single_feature_decoder()
    test_video_feature_decoder()
    print("Decoder 测试通过。")


if __name__ == "__main__":
    main()