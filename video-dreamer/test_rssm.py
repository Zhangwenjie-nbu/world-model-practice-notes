import torch

from models.rssm import RSSM, RSSMState


def check_state_shapes(
    state: RSSMState,
    batch_size: int,
    deter_dim: int,
    stoch_dim: int,
    has_posterior: bool,
):
    """
    检查 RSSMState 中各个张量的 shape 是否正确。
    """
    assert state.deter.shape == (batch_size, deter_dim), (
        f"deter shape 错误，期望 {(batch_size, deter_dim)}，实际得到 {state.deter.shape}"
    )

    assert state.stoch.shape == (batch_size, stoch_dim), (
        f"stoch shape 错误，期望 {(batch_size, stoch_dim)}，实际得到 {state.stoch.shape}"
    )

    assert state.prior_mean.shape == (batch_size, stoch_dim), (
        f"prior_mean shape 错误，期望 {(batch_size, stoch_dim)}，实际得到 {state.prior_mean.shape}"
    )

    assert state.prior_std.shape == (batch_size, stoch_dim), (
        f"prior_std shape 错误，期望 {(batch_size, stoch_dim)}，实际得到 {state.prior_std.shape}"
    )

    assert torch.all(state.prior_std > 0), "prior_std 必须全部大于 0。"

    if has_posterior:
        assert state.post_mean is not None, "posterior 状态下 post_mean 不应该为 None。"
        assert state.post_std is not None, "posterior 状态下 post_std 不应该为 None。"

        assert state.post_mean.shape == (batch_size, stoch_dim), (
            f"post_mean shape 错误，期望 {(batch_size, stoch_dim)}，实际得到 {state.post_mean.shape}"
        )

        assert state.post_std.shape == (batch_size, stoch_dim), (
            f"post_std shape 错误，期望 {(batch_size, stoch_dim)}，实际得到 {state.post_std.shape}"
        )

        assert torch.all(state.post_std > 0), "post_std 必须全部大于 0。"

    else:
        assert state.post_mean is None, "imagine 状态下 post_mean 应该为 None。"
        assert state.post_std is None, "imagine 状态下 post_std 应该为 None。"


def test_init_state():
    """
    测试 RSSM 初始状态。
    """
    batch_size = 16
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

    state = rssm.init_state(batch_size=batch_size, device=device)

    check_state_shapes(
        state=state,
        batch_size=batch_size,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
        has_posterior=True,
    )

    print("init_state 测试通过。")


def test_observe_step():
    """
    测试单步 observe_step。
    """
    batch_size = 16
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

    prev_state = rssm.init_state(batch_size=batch_size, device=device)

    embed_t = torch.randn(batch_size, embedding_dim)

    state = rssm.observe_step(
        prev_state=prev_state,
        embed=embed_t,
    )

    check_state_shapes(
        state=state,
        batch_size=batch_size,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
        has_posterior=True,
    )

    feature = rssm.get_feature(state)

    expected_feature_shape = (batch_size, deter_dim + stoch_dim)

    assert feature.shape == expected_feature_shape, (
        f"feature shape 错误，期望 {expected_feature_shape}，实际得到 {feature.shape}"
    )

    print("observe_step 测试通过。")


def test_imagine_step():
    """
    测试单步 imagine_step。
    """
    batch_size = 16
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

    prev_state = rssm.init_state(batch_size=batch_size, device=device)

    state = rssm.imagine_step(prev_state=prev_state)

    check_state_shapes(
        state=state,
        batch_size=batch_size,
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
        has_posterior=False,
    )

    feature = rssm.get_feature(state)

    expected_feature_shape = (batch_size, deter_dim + stoch_dim)

    assert feature.shape == expected_feature_shape, (
        f"feature shape 错误，期望 {expected_feature_shape}，实际得到 {feature.shape}"
    )

    print("imagine_step 测试通过。")


def test_multiple_steps():
    """
    测试 RSSM 是否能连续递推多个时间步。
    """
    batch_size = 16
    seq_len = 20
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

    embeds = torch.randn(batch_size, seq_len, embedding_dim)

    state = rssm.init_state(batch_size=batch_size, device=device)

    states = []
    features = []

    for t in range(seq_len):
        embed_t = embeds[:, t]
        state = rssm.observe_step(state, embed_t)

        states.append(state)
        features.append(rssm.get_feature(state))

    features = torch.stack(features, dim=1)

    expected_features_shape = (batch_size, seq_len, deter_dim + stoch_dim)

    assert features.shape == expected_features_shape, (
        f"多步 feature shape 错误，期望 {expected_features_shape}，实际得到 {features.shape}"
    )

    print("多步 observe 递推测试通过。")
    print("features shape:", features.shape)


def main():
    test_init_state()
    test_observe_step()
    test_imagine_step()
    test_multiple_steps()
    print("RSSM 测试全部通过。")


if __name__ == "__main__":
    main()