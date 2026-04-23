import torch

from models.rssm import RSSM, RSSMRollout


def check_observe_rollout_shapes(
    rollout: RSSMRollout,
    batch_size: int,
    seq_len: int,
    deter_dim: int,
    stoch_dim: int,
):
    """
    检查 observe_rollout 的输出 shape 是否正确。
    """
    feature_dim = deter_dim + stoch_dim

    assert len(rollout.states) == seq_len, (
        f"states 长度错误，期望 {seq_len}，实际得到 {len(rollout.states)}"
    )

    assert rollout.features.shape == (batch_size, seq_len, feature_dim), (
        f"features shape 错误，期望 {(batch_size, seq_len, feature_dim)}，"
        f"实际得到 {rollout.features.shape}"
    )

    assert rollout.prior_means.shape == (batch_size, seq_len, stoch_dim), (
        f"prior_means shape 错误，期望 {(batch_size, seq_len, stoch_dim)}，"
        f"实际得到 {rollout.prior_means.shape}"
    )

    assert rollout.prior_stds.shape == (batch_size, seq_len, stoch_dim), (
        f"prior_stds shape 错误，期望 {(batch_size, seq_len, stoch_dim)}，"
        f"实际得到 {rollout.prior_stds.shape}"
    )

    assert rollout.post_means is not None, "observe_rollout 的 post_means 不应该为 None。"
    assert rollout.post_stds is not None, "observe_rollout 的 post_stds 不应该为 None。"

    assert rollout.post_means.shape == (batch_size, seq_len, stoch_dim), (
        f"post_means shape 错误，期望 {(batch_size, seq_len, stoch_dim)}，"
        f"实际得到 {rollout.post_means.shape}"
    )

    assert rollout.post_stds.shape == (batch_size, seq_len, stoch_dim), (
        f"post_stds shape 错误，期望 {(batch_size, seq_len, stoch_dim)}，"
        f"实际得到 {rollout.post_stds.shape}"
    )

    assert torch.all(rollout.prior_stds > 0), "prior_stds 必须全部大于 0。"
    assert torch.all(rollout.post_stds > 0), "post_stds 必须全部大于 0。"

    assert rollout.final_state is rollout.states[-1], (
        "final_state 应该等于 states 的最后一个元素。"
    )


def check_imagine_rollout_shapes(
    rollout: RSSMRollout,
    batch_size: int,
    horizon: int,
    deter_dim: int,
    stoch_dim: int,
):
    """
    检查 imagine_rollout 的输出 shape 是否正确。
    """
    feature_dim = deter_dim + stoch_dim

    assert len(rollout.states) == horizon, (
        f"states 长度错误，期望 {horizon}，实际得到 {len(rollout.states)}"
    )

    assert rollout.features.shape == (batch_size, horizon, feature_dim), (
        f"features shape 错误，期望 {(batch_size, horizon, feature_dim)}，"
        f"实际得到 {rollout.features.shape}"
    )

    assert rollout.prior_means.shape == (batch_size, horizon, stoch_dim), (
        f"prior_means shape 错误，期望 {(batch_size, horizon, stoch_dim)}，"
        f"实际得到 {rollout.prior_means.shape}"
    )

    assert rollout.prior_stds.shape == (batch_size, horizon, stoch_dim), (
        f"prior_stds shape 错误，期望 {(batch_size, horizon, stoch_dim)}，"
        f"实际得到 {rollout.prior_stds.shape}"
    )

    assert rollout.post_means is None, "imagine_rollout 的 post_means 应该为 None。"
    assert rollout.post_stds is None, "imagine_rollout 的 post_stds 应该为 None。"

    assert torch.all(rollout.prior_stds > 0), "prior_stds 必须全部大于 0。"

    assert rollout.final_state is rollout.states[-1], (
        "final_state 应该等于 states 的最后一个元素。"
    )


def test_observe_rollout():
    """
    测试完整 observe_rollout。
    """
    batch_size = 16
    seq_len = 20
    embedding_dim = 128
    deter_dim = 128
    stoch_dim = 32
    hidden_dim = 256

    rssm = RSSM(
        embedding_dim=embedding_dim,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
        hidden_dim=hidden_dim,
    )

    embeds = torch.randn(batch_size, seq_len, embedding_dim)

    rollout = rssm.observe_rollout(embeds)

    check_observe_rollout_shapes(
        rollout=rollout,
        batch_size=batch_size,
        seq_len=seq_len,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
    )

    print("observe_rollout 测试通过。")
    print("features shape:", rollout.features.shape)
    print("prior_means shape:", rollout.prior_means.shape)
    print("post_means shape:", rollout.post_means.shape)


def test_imagine_rollout_from_init_state():
    """
    测试从初始状态开始 imagine_rollout。
    """
    batch_size = 16
    horizon = 10
    embedding_dim = 128
    deter_dim = 128
    stoch_dim = 32
    hidden_dim = 256

    device = torch.device("cpu")

    rssm = RSSM(
        embedding_dim=embedding_dim,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
        hidden_dim=hidden_dim,
    )

    start_state = rssm.init_state(batch_size=batch_size, device=device)

    rollout = rssm.imagine_rollout(
        start_state=start_state,
        horizon=horizon,
    )

    check_imagine_rollout_shapes(
        rollout=rollout,
        batch_size=batch_size,
        horizon=horizon,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
    )

    print("从初始状态 imagine_rollout 测试通过。")
    print("features shape:", rollout.features.shape)


def test_context_observe_then_future_imagine():
    """
    测试真实项目中最重要的流程：
        先用 context embedding 做 observe_rollout，
        再从 context 的 final_state 出发做 future imagine_rollout。
    """
    batch_size = 16
    context_len = 10
    pred_len = 10
    embedding_dim = 128
    deter_dim = 128
    stoch_dim = 32
    hidden_dim = 256

    rssm = RSSM(
        embedding_dim=embedding_dim,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
        hidden_dim=hidden_dim,
    )

    context_embeds = torch.randn(batch_size, context_len, embedding_dim)

    context_rollout = rssm.observe_rollout(context_embeds)

    future_rollout = rssm.imagine_rollout(
        start_state=context_rollout.final_state,
        horizon=pred_len,
    )

    check_observe_rollout_shapes(
        rollout=context_rollout,
        batch_size=batch_size,
        seq_len=context_len,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
    )

    check_imagine_rollout_shapes(
        rollout=future_rollout,
        batch_size=batch_size,
        horizon=pred_len,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
    )

    print("context observe + future imagine 测试通过。")
    print("context features shape:", context_rollout.features.shape)
    print("future features shape:", future_rollout.features.shape)


def test_wrong_embedding_dim():
    """
    测试 embedding_dim 不匹配时是否能正确报错。
    """
    batch_size = 16
    seq_len = 20
    embedding_dim = 128
    wrong_embedding_dim = 64

    rssm = RSSM(embedding_dim=embedding_dim)

    embeds = torch.randn(batch_size, seq_len, wrong_embedding_dim)

    try:
        _ = rssm.observe_rollout(embeds)
    except ValueError as e:
        print("embedding_dim 错误检查通过。")
        print("捕获到错误:", e)
        return

    raise AssertionError("embedding_dim 不匹配时应该抛出 ValueError，但没有抛出。")


def main():
    test_observe_rollout()
    test_imagine_rollout_from_init_state()
    test_context_observe_then_future_imagine()
    test_wrong_embedding_dim()
    print("RSSM rollout 测试全部通过。")


if __name__ == "__main__":
    main()