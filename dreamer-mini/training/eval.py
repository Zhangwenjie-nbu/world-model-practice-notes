import torch


@torch.no_grad()
def evaluate_actor(env, rssm, actor, num_episodes=5, max_steps=300, device="cpu"):
    """
    用当前 actor 在真实环境中评估若干条 episode。

    这里是 Dreamer-mini 的最小评估版：
    - 每一步根据当前 obs 更新 posterior state
    - 再用 actor 从 latent feature 里选动作
    - 不做训练，只统计真实环境 return

    返回:
        avg_return: 平均 episode return
    """
    rssm.eval()
    actor.eval()

    returns = []

    for _ in range(num_episodes):
        obs, _ = env.reset()

        # 初始化 latent state
        state = rssm.initial_state(batch_size=1)

        # t=0 时没有前一动作，补 0
        prev_action = torch.zeros(1, env.action_dim, device=device)

        episode_return = 0.0

        for _ in range(max_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # 用 observe_step 把当前观测并入 latent
            state, _, _, _, _ = rssm.observe_step(state, prev_action, obs_t)

            feat = rssm.get_feat(state)   # [1, feat_dim]
            action, _, _ = actor(feat)    # [1, action_dim]

            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, done, _ = env.step(action_np)

            episode_return += reward
            obs = next_obs

            prev_action = action

            if done:
                break

        returns.append(episode_return)

    avg_return = sum(returns) / len(returns)
    return avg_return