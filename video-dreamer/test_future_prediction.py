import torch

from models.world_model import WorldModel


def main():
    """
    测试 WorldModel 的未来预测接口。
    """
    batch_size = 8
    context_len = 10
    pred_len = 10
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

    context_video = torch.rand(batch_size, context_len, channels, image_size, image_size)

    output = model.predict_future(
        context_video=context_video,
        pred_len=pred_len,
    )

    print("context_embeddings shape:", output.context_embeddings.shape)
    print("context_rollout features shape:", output.context_rollout.features.shape)
    print("future_rollout features shape:", output.future_rollout.features.shape)
    print("future_predictions shape:", output.future_predictions.shape)

    assert output.context_embeddings.shape == (batch_size, context_len, 128)
    assert output.context_rollout.features.shape == (batch_size, context_len, 160)
    assert output.future_rollout.features.shape == (batch_size, pred_len, 160)
    assert output.future_predictions.shape == (batch_size, pred_len, channels, image_size, image_size)

    print("未来帧预测接口测试通过。")


if __name__ == "__main__":
    main()