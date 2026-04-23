import torch
import torch.nn as nn


class ImageDecoder(nn.Module):
    """
    图像解码器。

    作用：
        将 RSSM 的 latent feature 解码成图像。

    输入：
        feature: [B, feature_dim]

    输出：
        recon: [B, C, H, W]

    在当前项目中：
        feature_dim = deter_dim + stoch_dim = 128 + 32 = 160
        C = 1
        H = W = 64
    """

    def __init__(
        self,
        feature_dim: int = 160,
        out_channels: int = 1,
        image_size: int = 64,
        base_channels: int = 128,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.out_channels = out_channels
        self.image_size = image_size
        self.base_channels = base_channels

        # 当前 Decoder 设计为：
        # feature -> 线性层 -> [B, base_channels, 4, 4]
        # 然后通过反卷积逐步上采样：
        # 4 -> 8 -> 16 -> 32 -> 64
        #
        # 因此要求 image_size 能够被 16 整除。
        if image_size % 16 != 0:
            raise ValueError(
                f"image_size={image_size} 必须能被 16 整除，当前 Decoder 才能进行 4 次 2 倍上采样。"
            )

        self.init_spatial_size = image_size // 16

        # 将一维 latent feature 映射成小尺寸 feature map
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=feature_dim,
                out_features=base_channels * self.init_spatial_size * self.init_spatial_size,
            ),
            nn.LayerNorm(base_channels * self.init_spatial_size * self.init_spatial_size),
            nn.ReLU(inplace=True),
        )

        # 通过 ConvTranspose2d 逐步上采样回图像空间
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=base_channels,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),

            # Moving MNIST 的像素范围是 [0, 1]，因此最后用 Sigmoid 约束输出范围
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            feature: latent feature，shape 为 [B, feature_dim]

        返回：
            recon: 重建图像，shape 为 [B, out_channels, image_size, image_size]
        """
        if feature.ndim != 2:
            raise ValueError(
                f"ImageDecoder 期望输入 shape 为 [B, feature_dim]，但得到 {feature.shape}"
            )

        B = feature.shape[0]

        x = self.fc(feature)  # [B, base_channels * init_spatial_size * init_spatial_size]

        x = x.reshape(
            B,
            self.base_channels,
            self.init_spatial_size,
            self.init_spatial_size,
        )  # [B, base_channels, 4, 4]

        recon = self.deconv(x)  # [B, out_channels, image_size, image_size]

        return recon


def decode_video_features(
    decoder: ImageDecoder,
    features: torch.Tensor,
) -> torch.Tensor:
    """
    对一整个视频 latent feature 序列进行解码。

    参数：
        decoder:
            ImageDecoder 实例。

        features:
            latent feature 序列，shape 为 [B, T, feature_dim]。

    返回：
        recons:
            重建视频，shape 为 [B, T, C, H, W]。

    说明：
        Decoder 本身只处理 [B, feature_dim]。
        为了高效处理整个序列，这里先把 B 和 T 合并：
            [B, T, D] -> [B*T, D]
        解码后再 reshape 回：
            [B*T, C, H, W] -> [B, T, C, H, W]
    """
    if features.ndim != 3:
        raise ValueError(
            f"decode_video_features 期望输入 shape 为 [B, T, feature_dim]，但得到 {features.shape}"
        )

    B, T, D = features.shape

    features_flat = features.reshape(B * T, D)     # [B*T, D]
    recons_flat = decoder(features_flat)           # [B*T, C, H, W]
    C, H, W = recons_flat.shape[1], recons_flat.shape[2], recons_flat.shape[3]

    recons = recons_flat.reshape(B, T, C, H, W)    # [B, T, C, H, W]

    return recons