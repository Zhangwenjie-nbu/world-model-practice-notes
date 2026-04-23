import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """
    图像编码器。

    作用：
        将单帧图像 x_t 编码成一个向量 embedding e_t。

    输入：
        x: [B, C, H, W]

    输出：
        embedding: [B, embedding_dim]

    在当前项目中：
        C = 1
        H = W = 64
        embedding_dim = 128
    """

    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 128,
        image_size: int = 64,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.image_size = image_size

        # 对 64x64 图像进行逐步下采样：
        # 64 -> 32 -> 16 -> 8 -> 4
        #
        # 输入:  [B, 1, 64, 64]
        # 输出:  [B, 128, 4, 4]
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # 计算卷积后的 feature map 尺寸
        conv_out_size = image_size // 16

        if conv_out_size <= 0:
            raise ValueError(
                f"image_size={image_size} 太小，经过 4 次 stride=2 下采样后尺寸非法。"
            )

        self.conv_out_dim = 128 * conv_out_size * conv_out_size

        # 将卷积特征展平后映射到 embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(self.conv_out_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            x: 单帧图像 batch，shape 为 [B, C, H, W]

        返回：
            embedding: 图像 embedding，shape 为 [B, embedding_dim]
        """
        if x.ndim != 4:
            raise ValueError(
                f"ImageEncoder 期望输入 shape 为 [B, C, H, W]，但得到 {x.shape}"
            )

        feat = self.conv(x)              # [B, 128, H/16, W/16]
        feat = feat.flatten(start_dim=1) # [B, conv_out_dim]
        embedding = self.fc(feat)        # [B, embedding_dim]

        return embedding


def encode_video_frames(
    encoder: ImageEncoder,
    video: torch.Tensor,
) -> torch.Tensor:
    """
    对一整个视频 batch 的每一帧进行编码。

    参数：
        encoder:
            ImageEncoder 实例。

        video:
            视频 batch，shape 为 [B, T, C, H, W]。

    返回：
        embeddings:
            每一帧对应的 embedding，shape 为 [B, T, embedding_dim]。

    说明：
        Encoder 本身只处理 [B, C, H, W]。
        为了高效处理整个视频，这里先把 B 和 T 合并：
            [B, T, C, H, W] -> [B*T, C, H, W]
        编码后再 reshape 回：
            [B*T, D] -> [B, T, D]
    """
    if video.ndim != 5:
        raise ValueError(
            f"encode_video_frames 期望输入 shape 为 [B, T, C, H, W]，但得到 {video.shape}"
        )

    B, T, C, H, W = video.shape

    video_flat = video.reshape(B * T, C, H, W)  # [B*T, C, H, W]
    embeddings_flat = encoder(video_flat)       # [B*T, D]
    embeddings = embeddings_flat.reshape(B, T, -1)  # [B, T, D]

    return embeddings