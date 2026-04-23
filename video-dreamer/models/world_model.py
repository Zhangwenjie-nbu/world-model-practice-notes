from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.encoder import ImageEncoder, encode_video_frames
from models.decoder import ImageDecoder, decode_video_features
from models.rssm import RSSM, RSSMRollout, RSSMState


@dataclass
class WorldModelForwardOutput:
    """
    WorldModel 整段 observe 前向输出结构。

    字段说明：
        embeddings:
            Encoder 输出的观测 embedding 序列，shape 为 [B, T, embedding_dim]

        rollout:
            RSSM observe_rollout 的结果。

        reconstructions:
            Decoder 输出的整段重建视频，shape 为 [B, T, C, H, W]
    """

    embeddings: torch.Tensor
    rollout: RSSMRollout
    reconstructions: torch.Tensor


@dataclass
class FuturePredictionOutput:
    """
    未来帧预测输出结构。

    字段说明：
        context_embeddings:
            前 K 帧的 embedding，shape 为 [B, K, embedding_dim]

        context_rollout:
            context 的 observe_rollout 结果。

        future_rollout:
            从 context final_state 出发做 imagine_rollout 的结果。

        future_predictions:
            未来 H 帧预测图像，shape 为 [B, H, C, H_img, W_img]
    """

    context_embeddings: torch.Tensor
    context_rollout: RSSMRollout
    future_rollout: RSSMRollout
    future_predictions: torch.Tensor


class WorldModel(nn.Module):
    """
    RSSM 视频世界模型总封装。
    """

    def __init__(
        self,
        image_channels: int = 1,
        image_size: int = 64,
        embedding_dim: int = 128,
        deter_dim: int = 128,
        stoch_dim: int = 32,
        hidden_dim: int = 256,
        min_std: float = 0.1,
    ):
        super().__init__()

        self.image_channels = image_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.feature_dim = deter_dim + stoch_dim

        self.encoder = ImageEncoder(
            in_channels=image_channels,
            embedding_dim=embedding_dim,
            image_size=image_size,
        )

        self.rssm = RSSM(
            embedding_dim=embedding_dim,
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            hidden_dim=hidden_dim,
            min_std=min_std,
        )

        self.decoder = ImageDecoder(
            feature_dim=self.feature_dim,
            out_channels=image_channels,
            image_size=image_size,
        )

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        对整段视频进行编码。

        参数：
            video: [B, T, C, H, W]

        返回：
            embeddings: [B, T, embedding_dim]
        """
        embeddings = encode_video_frames(self.encoder, video)
        return embeddings

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        对整段 latent feature 序列进行解码。

        参数：
            features: [B, T, feature_dim]

        返回：
            reconstructions: [B, T, C, H, W]
        """
        reconstructions = decode_video_features(self.decoder, features)
        return reconstructions

    def forward_observe(
        self,
        video: torch.Tensor,
        init_state: Optional[RSSMState] = None,
    ) -> WorldModelForwardOutput:
        """
        对真实视频执行完整 observe 前向过程。
        """
        if video.ndim != 5:
            raise ValueError(
                f"forward_observe 期望 video shape 为 [B, T, C, H, W]，但得到 {video.shape}"
            )

        embeddings = self.encode(video)
        rollout = self.rssm.observe_rollout(
            embeds=embeddings,
            init_state=init_state,
        )
        reconstructions = self.decode(rollout.features)

        return WorldModelForwardOutput(
            embeddings=embeddings,
            rollout=rollout,
            reconstructions=reconstructions,
        )

    def reconstruct(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """
        对输入视频做整段重建。
        """
        output = self.forward_observe(video)
        return output.reconstructions

    def predict_future(
        self,
        context_video: torch.Tensor,
        pred_len: int,
    ) -> FuturePredictionOutput:
        """
        给定前 K 帧上下文，预测未来 pred_len 帧。

        流程：
            context_video
                -> encoder
                -> context embeddings
                -> rssm.observe_rollout
                -> context final_state
                -> rssm.imagine_rollout(horizon=pred_len)
                -> decoder
                -> future_predictions

        参数：
            context_video:
                前 K 帧真实视频，shape 为 [B, K, C, H, W]

            pred_len:
                需要预测的未来帧数 H。

        返回：
            FuturePredictionOutput
        """
        if context_video.ndim != 5:
            raise ValueError(
                f"predict_future 期望 context_video shape 为 [B, K, C, H, W]，但得到 {context_video.shape}"
            )

        if pred_len <= 0:
            raise ValueError(f"pred_len 必须大于 0，但得到 {pred_len}")

        # 1. 编码 context
        context_embeddings = self.encode(context_video)  # [B, K, embedding_dim]

        # 2. 用真实 context 做 observe rollout
        context_rollout = self.rssm.observe_rollout(
            embeds=context_embeddings,
            init_state=None,
        )

        # 3. 从 context 的最后状态出发，向未来 imagine
        future_rollout = self.rssm.imagine_rollout(
            start_state=context_rollout.final_state,
            horizon=pred_len,
        )

        # 4. 解码未来 latent feature
        future_predictions = self.decode(future_rollout.features)  # [B, H, C, H, W]

        return FuturePredictionOutput(
            context_embeddings=context_embeddings,
            context_rollout=context_rollout,
            future_rollout=future_rollout,
            future_predictions=future_predictions,
        )