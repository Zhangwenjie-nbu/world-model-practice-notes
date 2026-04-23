import torch


def imagine_rollout(rssm, actor, reward_head,
                     start_states, horizon):
    """
    在 latent space 中做 imagination rollout

    输入:
        rssm
        actor
        reward_head
        start_states: list of RSSMState（长度 T）
        horizon: rollout 步数 H

    输出:
        imag_feats:   [B, H, feat_dim]
        imag_actions: [B, H, action_dim]
        imag_rewards: [B, H, 1]
    """

    # ===== 1. 取最后一个 state 作为起点 =====
    state = start_states[-1]

    imag_feats = []
    imag_actions = []
    imag_rewards = []

    for _ in range(horizon):

        # 当前 feature
        feat = torch.cat([state.h, state.z], dim=-1)

        # ===== 2. actor 生成 action =====
        action, _, _ = actor(feat)

        # ===== 3. 用 prior 做一步 rollout =====
        state, _, _ = rssm.img_step(state, action)

        # 新 feature
        next_feat = torch.cat([state.h, state.z], dim=-1)

        # ===== 4. 预测 reward =====
        reward = reward_head(next_feat)

        # ===== 5. 存储 =====
        imag_feats.append(next_feat)
        imag_actions.append(action)
        imag_rewards.append(reward)

    # ===== stack =====
    imag_feats = torch.stack(imag_feats, dim=1)
    imag_actions = torch.stack(imag_actions, dim=1)
    imag_rewards = torch.stack(imag_rewards, dim=1)

    return imag_feats, imag_actions, imag_rewards