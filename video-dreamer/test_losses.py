import torch

from training.losses import (
    reconstruction_loss,
    gaussian_kl_divergence,
    apply_free_nats,
    world_model_loss,
)


def test_reconstruction_loss_mse():
    """
    测试 MSE 重建损失。
    """
    batch_size = 16
    seq_len = 20
    channels = 1
    image_size = 64

    recon = torch.rand(batch_size, seq_len, channels, image_size, image_size)
    target = torch.rand(batch_size, seq_len, channels, image_size, image_size)

    loss = reconstruction_loss(
        recon=recon,
        target=target,
        loss_type="mse",
    )

    assert loss.ndim == 0, f"重建损失应该是标量，但得到 shape {loss.shape}"
    assert loss.item() >= 0.0, "MSE 重建损失应该大于等于 0。"

    print("MSE reconstruction loss 测试通过。")
    print("mse loss:", loss.item())


def test_reconstruction_loss_bce():
    """
    测试 BCE 重建损失。
    """
    batch_size = 16
    seq_len = 20
    channels = 1
    image_size = 64

    recon = torch.rand(batch_size, seq_len, channels, image_size, image_size)
    target = torch.rand(batch_size, seq_len, channels, image_size, image_size)

    loss = reconstruction_loss(
        recon=recon,
        target=target,
        loss_type="bce",
    )

    assert loss.ndim == 0, f"BCE 损失应该是标量，但得到 shape {loss.shape}"
    assert loss.item() >= 0.0, "BCE 损失应该大于等于 0。"

    print("BCE reconstruction loss 测试通过。")
    print("bce loss:", loss.item())


def test_gaussian_kl_same_distribution():
    """
    测试当 posterior 和 prior 完全相同时，KL 应该接近 0。
    """
    batch_size = 16
    seq_len = 20
    stoch_dim = 32

    mean = torch.zeros(batch_size, seq_len, stoch_dim)
    std = torch.ones(batch_size, seq_len, stoch_dim)

    kl = gaussian_kl_divergence(
        post_mean=mean,
        post_std=std,
        prior_mean=mean,
        prior_std=std,
        reduce=True,
    )

    assert kl.ndim == 0, f"KL 应该是标量，但得到 shape {kl.shape}"
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-5), (
        f"相同分布的 KL 应该接近 0，但得到 {kl.item()}"
    )

    print("相同高斯分布 KL 测试通过。")
    print("kl:", kl.item())


def test_gaussian_kl_different_distribution():
    """
    测试当 posterior 和 prior 不同时，KL 应该大于 0。
    """
    batch_size = 16
    seq_len = 20
    stoch_dim = 32

    post_mean = torch.ones(batch_size, seq_len, stoch_dim)
    post_std = torch.ones(batch_size, seq_len, stoch_dim)

    prior_mean = torch.zeros(batch_size, seq_len, stoch_dim)
    prior_std = torch.ones(batch_size, seq_len, stoch_dim)

    kl = gaussian_kl_divergence(
        post_mean=post_mean,
        post_std=post_std,
        prior_mean=prior_mean,
        prior_std=prior_std,
        reduce=True,
    )

    assert kl.ndim == 0, f"KL 应该是标量，但得到 shape {kl.shape}"
    assert kl.item() > 0.0, "不同分布的 KL 应该大于 0。"

    print("不同高斯分布 KL 测试通过。")
    print("kl:", kl.item())


def test_gaussian_kl_no_reduce():
    """
    测试 reduce=False 时，KL 输出 shape 是否为 [B, T]。
    """
    batch_size = 16
    seq_len = 20
    stoch_dim = 32

    post_mean = torch.randn(batch_size, seq_len, stoch_dim)
    post_std = torch.ones(batch_size, seq_len, stoch_dim)

    prior_mean = torch.zeros(batch_size, seq_len, stoch_dim)
    prior_std = torch.ones(batch_size, seq_len, stoch_dim)

    kl = gaussian_kl_divergence(
        post_mean=post_mean,
        post_std=post_std,
        prior_mean=prior_mean,
        prior_std=prior_std,
        reduce=False,
    )

    expected_shape = (batch_size, seq_len)

    assert kl.shape == expected_shape, (
        f"reduce=False 时 KL shape 错误，期望 {expected_shape}，实际得到 {kl.shape}"
    )
    assert torch.all(kl >= 0.0), "KL 每个元素都应该大于等于 0。"

    print("KL reduce=False 测试通过。")
    print("kl shape:", kl.shape)


def test_apply_free_nats():
    """
    测试 free_nats 裁剪逻辑。
    """
    low_kl = torch.tensor(0.3)
    high_kl = torch.tensor(2.5)
    free_nats = 1.0

    low_result = apply_free_nats(low_kl, free_nats)
    high_result = apply_free_nats(high_kl, free_nats)

    assert torch.allclose(low_result, torch.tensor(1.0)), (
        f"低 KL 应该被裁剪到 1.0，但得到 {low_result.item()}"
    )

    assert torch.allclose(high_result, torch.tensor(2.5)), (
        f"高 KL 不应该被裁剪，但得到 {high_result.item()}"
    )

    print("free_nats 测试通过。")
    print("low_result:", low_result.item())
    print("high_result:", high_result.item())


def test_world_model_loss():
    """
    测试完整 world model loss。
    """
    batch_size = 16
    seq_len = 20
    channels = 1
    image_size = 64
    stoch_dim = 32

    recon = torch.rand(batch_size, seq_len, channels, image_size, image_size)
    target = torch.rand(batch_size, seq_len, channels, image_size, image_size)

    prior_mean = torch.zeros(batch_size, seq_len, stoch_dim)
    prior_std = torch.ones(batch_size, seq_len, stoch_dim)

    post_mean = torch.randn(batch_size, seq_len, stoch_dim)
    post_std = torch.ones(batch_size, seq_len, stoch_dim)

    output = world_model_loss(
        recon=recon,
        target=target,
        prior_mean=prior_mean,
        prior_std=prior_std,
        post_mean=post_mean,
        post_std=post_std,
        recon_loss_type="mse",
        kl_weight=1.0,
        free_nats=1.0,
    )

    assert output.total_loss.ndim == 0, "total_loss 应该是标量。"
    assert output.recon_loss.ndim == 0, "recon_loss 应该是标量。"
    assert output.kl_loss.ndim == 0, "kl_loss 应该是标量。"
    assert output.raw_kl_loss.ndim == 0, "raw_kl_loss 应该是标量。"

    assert output.total_loss.item() >= 0.0, "total_loss 应该大于等于 0。"
    assert output.recon_loss.item() >= 0.0, "recon_loss 应该大于等于 0。"
    assert output.kl_loss.item() >= 0.0, "kl_loss 应该大于等于 0。"
    assert output.raw_kl_loss.item() >= 0.0, "raw_kl_loss 应该大于等于 0。"

    required_keys = [
        "total_loss",
        "recon_loss",
        "kl_loss",
        "raw_kl_loss",
        "kl_weight",
        "free_nats",
    ]

    for key in required_keys:
        assert key in output.loss_dict, f"loss_dict 中缺少字段: {key}"

    print("world_model_loss 测试通过。")
    print(output.loss_dict)


def test_shape_mismatch():
    """
    测试重建图像和真实图像 shape 不一致时是否会报错。
    """
    recon = torch.rand(16, 20, 1, 64, 64)
    target = torch.rand(16, 10, 1, 64, 64)

    try:
        _ = reconstruction_loss(recon, target)
    except ValueError as e:
        print("shape mismatch 检查通过。")
        print("捕获到错误:", e)
        return

    raise AssertionError("shape 不匹配时应该抛出 ValueError，但没有抛出。")


def main():
    test_reconstruction_loss_mse()
    test_reconstruction_loss_bce()
    test_gaussian_kl_same_distribution()
    test_gaussian_kl_different_distribution()
    test_gaussian_kl_no_reduce()
    test_apply_free_nats()
    test_world_model_loss()
    test_shape_mismatch()
    print("losses 测试全部通过。")


if __name__ == "__main__":
    main()