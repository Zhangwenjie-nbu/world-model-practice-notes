import os
import torch


def save_checkpoint(path, rssm, obs_head, reward_head, actor, value_net,
                    wm_optimizer, actor_optimizer, value_optimizer, step):
    """
    保存训练检查点。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "step": step,
        "rssm": rssm.state_dict(),
        "obs_head": obs_head.state_dict(),
        "reward_head": reward_head.state_dict(),
        "actor": actor.state_dict(),
        "value_net": value_net.state_dict(),
        "wm_optimizer": wm_optimizer.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "value_optimizer": value_optimizer.state_dict(),
    }, path)


def load_checkpoint(path, rssm, obs_head, reward_head, actor, value_net,
                    wm_optimizer=None, actor_optimizer=None, value_optimizer=None,
                    map_location="cpu"):
    """
    加载训练检查点。
    """
    ckpt = torch.load(path, map_location=map_location)

    rssm.load_state_dict(ckpt["rssm"])
    obs_head.load_state_dict(ckpt["obs_head"])
    reward_head.load_state_dict(ckpt["reward_head"])
    actor.load_state_dict(ckpt["actor"])
    value_net.load_state_dict(ckpt["value_net"])

    if wm_optimizer is not None:
        wm_optimizer.load_state_dict(ckpt["wm_optimizer"])
    if actor_optimizer is not None:
        actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
    if value_optimizer is not None:
        value_optimizer.load_state_dict(ckpt["value_optimizer"])

    return ckpt["step"]