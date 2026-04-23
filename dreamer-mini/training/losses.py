import torch
import torch.nn.functional as F


def kl_divergence_free(mean_q, std_q, mean_p, std_p, free_bits=1.0):
    """
    计算带有 free bits 的 KL 散度

    输入：
        mean_q, std_q: posterior 的分布参数
        mean_p, std_p: prior 的分布参数
        free_bits: KL 的最小下限，防止 KL 变得过小

    输出：
        标量 KL loss
    """
    var_q = std_q ** 2
    var_p = std_p ** 2

    kl = torch.log(std_p / std_q) + (var_q + (mean_q - mean_p) ** 2) / (2 * var_p) - 0.5

    # 限制 KL 最小值，防止 KL collapse
    kl = torch.clamp(kl, min=free_bits)

    return kl.mean()


def kl_divergence_balance(mean_q, std_q, mean_p, std_p, alpha=0.8):
    """
    DreamerV2 风格的 KL balance

    输入：
        mean_q, std_q: posterior 的分布参数
        mean_p, std_p: prior 的分布参数
        alpha: balance 系数

    输出：
        标量 KL loss
    """
    def kl(m1, s1, m2, s2):
        var1 = s1 ** 2
        var2 = s2 ** 2
        return torch.log(s2 / s1) + (var1 + (m1 - m2) ** 2) / (2 * var2) - 0.5

    # 计算两部分 KL loss
    kl_q = kl(mean_q, std_q, mean_p.detach(), std_p.detach())
    kl_p = kl(mean_q.detach(), std_q.detach(), mean_p, std_p)

    # 加权平衡两部分的 KL
    kl_total = alpha * kl_q + (1 - alpha) * kl_p

    return kl_total.mean()


def kl_loss_final(mean_q, std_q, mean_p, std_p, free_bits=1.0, alpha=0.8):
    """
    最终的 KL loss，结合了 free bits 和 KL balance

    输入：
        mean_q, std_q: posterior 的分布参数
        mean_p, std_p: prior 的分布参数
        free_bits: KL 的最小下限，防止 KL collapse
        alpha: balance 系数

    输出：
        标量 KL loss
    """
    def kl(m1, s1, m2, s2):
        var1 = s1 ** 2
        var2 = s2 ** 2
        return torch.log(s2 / s1) + (var1 + (m1 - m2) ** 2) / (2 * var2) - 0.5

    # 计算两部分 KL loss
    kl_q = kl(mean_q, std_q, mean_p.detach(), std_p.detach())
    kl_p = kl(mean_q.detach(), std_q.detach(), mean_p, std_p)

    # 加权平衡两部分的 KL
    kl = alpha * kl_q + (1 - alpha) * kl_p

    # free bits：避免 KL 过小，导致 latent 失去信息
    kl = torch.clamp(kl, min=free_bits)

    return kl.mean()


def world_model_loss(
    rssm,
    obs_head,
    reward_head,
    obs,
    actions,
    rewards,
    kl_loss_fn,
):
    """
    计算 world model 的完整损失。

    输入:
        obs:      [B, T, obs_dim]
        actions:  [B, T, action_dim]
        rewards:  [B, T]

    返回:
        total_loss: 标量
        metrics:    字典，方便打印
        outputs:    中间结果，供后续 imagination 使用
    """
    # 1. RSSM rollout
    states, p_mean, p_std, q_mean, q_std = rssm.observe_rollout(obs, actions)

    # 2. latent feature
    feat = rssm.get_feat_seq(states)  # [B, T, feat_dim]

    # 3. heads prediction
    obs_pred = obs_head(feat)                     # [B, T, obs_dim]
    reward_pred = reward_head(feat).squeeze(-1)  # [B, T]

    # 4. losses
    obs_loss = ((obs_pred - obs) ** 2).mean()
    reward_loss = ((reward_pred - rewards) ** 2).mean()
    kl_loss = kl_loss_fn(q_mean, q_std, p_mean, p_std)

    total_loss = obs_loss + reward_loss + kl_loss

    metrics = {
        "wm_loss": total_loss.item(),
        "obs_loss": obs_loss.item(),
        "reward_loss": reward_loss.item(),
        "kl_loss": kl_loss.item(),
    }

    outputs = {
        "states": states,
        "feat": feat,
        "prior_mean": p_mean,
        "prior_std": p_std,
        "post_mean": q_mean,
        "post_std": q_std,
    }

    return total_loss, metrics, outputs


def compute_lambda_return(rewards, values, gamma=0.99, lambda_=0.95):
    """
    计算 λ-return，用于策略优化。

    输入：
        rewards: [B, T]，奖励序列
        values: [B, T]，价值估计序列
        gamma: 折扣因子
        lambda_: λ-return 参数

    输出：
        lambda_returns: [B, T]，λ-return 序列
    """
    if rewards.dim() == 3 and rewards.size(-1) == 1:
        rewards = rewards.squeeze(-1)
    if values.dim() == 3 and values.size(-1) == 1:
        values = values.squeeze(-1)

    returns = torch.zeros_like(rewards)
    bootstrap = values[:, -1]
    next_return = bootstrap

    for t in reversed(range(rewards.size(1))):
        next_value = values[:, t + 1] if t + 1 < values.size(1) else bootstrap
        next_return = rewards[:, t] + gamma * (
            (1.0 - lambda_) * next_value + lambda_ * next_return
        )
        returns[:, t] = next_return

    return returns


def actor_loss(lambda_returns):
    """
    Actor 的最小教学版 loss：
    直接最大化 imagined lambda-return

    输入：
        lambda_returns: [B, H]（lambda-return 序列）

    返回：
        loss: 标量
    """
    return -lambda_returns.mean()


def compute_entropy(actor, feat):
    """
    计算策略的熵（Entropy Bonus）

    输入：
        actor: 策略网络
        feat: [B, T, feat_dim]，latent 特征

    输出：
        entropy: 策略的熵
    """
    actions, mean, std = actor(feat)

    # 计算 log prob
    log_prob = torch.log(torch.clamp(actions, min=-1.0, max=1.0))

    # 计算熵
    entropy = -torch.mean(log_prob)

    return entropy


def actor_loss_with_entropy(lambda_returns, entropy=None, entropy_coeff=0.01):
    """
    计算最终的 Actor 损失（包括策略梯度和熵奖励）

    输入：
        lambda_returns: [B, H]，λ-return 序列
        entropy:        [B, H]，策略的熵（如果没有则为 None）
        entropy_coeff: 熵奖励系数

    输出：
        loss: Actor 最终损失
    """
    # 最大化 lambda_return
    loss = actor_loss(lambda_returns)

    if entropy is not None and entropy_coeff > 0.0:
        # 计算熵奖励，并加到损失中
        loss -= entropy_coeff * entropy

    return loss


def value_loss(value_net, feat, lambda_returns):
    """
    计算 Value 网络的损失（均方误差）

    输入：
        value_net: 价值网络
        feat: [B, T, feat_dim]，latent 特征
        lambda_returns: [B, T]，λ-return 序列

    输出：
        loss: Value 网络的损失
    """
    # 计算价值估计
    value_preds = value_net(feat)

    # 计算价值损失
    loss = F.mse_loss(value_preds.squeeze(-1), lambda_returns)

    return loss