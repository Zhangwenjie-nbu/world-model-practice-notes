import yaml
import torch

from utils.seed import set_seed
from envs.make_env import EnvWrapper
from training.collector import Collector, RandomPolicy
from data.replay_buffer import ReplayBuffer
from models.rssm import RSSM
from models.heads import ObsHead, RewardHead
from models.actor import Actor
from models.value import Value
from training.trainer import DreamerTrainer
from training.losses import kl_loss_final
from training.eval import evaluate_actor
from utils.logger import CSVLogger
from utils.checkpoint import save_checkpoint


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    logger = CSVLogger("logs/train_metrics.csv")
    cfg = load_config("configs/pendulum.yaml")
    set_seed(cfg["seed"])

    device = torch.device(
        "cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    )

    env = EnvWrapper(cfg["env"]["name"])
    cfg["model"]["obs_dim"] = env.obs_dim
    cfg["model"]["action_dim"] = env.action_dim

    # 采集器与 buffer
    policy = RandomPolicy(env)
    collector = Collector(env)
    replay_buffer = ReplayBuffer(
        capacity=cfg["buffer"]["capacity"],
        seq_len=cfg["train"]["seq_len"],
        device=device,
    )

    # 模型
    deter_dim = cfg["model"]["deter_dim"]
    stoch_dim = cfg["model"]["stoch_dim"]
    hidden_dim = cfg["model"]["hidden_dim"]
    obs_dim = cfg["model"]["obs_dim"]
    action_dim = cfg["model"]["action_dim"]
    feat_dim = deter_dim + stoch_dim

    rssm = RSSM(
        deter_dim=deter_dim,
        stoch_dim=stoch_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        obs_dim=obs_dim,
        device=device,
    ).to(device)

    obs_head = ObsHead(feat_dim, obs_dim, hidden_dim).to(device)
    reward_head = RewardHead(feat_dim, hidden_dim).to(device)
    actor = Actor(feat_dim, action_dim, hidden_dim).to(device)
    value_net = Value(feat_dim, hidden_dim).to(device)

    # optimizer
    wm_optimizer = torch.optim.Adam(
        list(rssm.parameters()) + list(obs_head.parameters()) + list(reward_head.parameters()),
        lr=float(cfg["optim"]["world_model_lr"]),
    )
    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=float(cfg["optim"]["actor_lr"]),
    )
    value_optimizer = torch.optim.Adam(
        value_net.parameters(),
        lr=float(cfg["optim"]["critic_lr"]),
    )

    trainer = DreamerTrainer(
        rssm=rssm,
        obs_head=obs_head,
        reward_head=reward_head,
        actor=actor,
        value_net=value_net,
        replay_buffer=replay_buffer,
        wm_optimizer=wm_optimizer,
        actor_optimizer=actor_optimizer,
        value_optimizer=value_optimizer,
        kl_loss_fn=kl_loss_final,
        imagine_horizon=cfg["rl"]["imagine_horizon"],
        gamma=cfg["rl"]["gamma"],
        lambda_=cfg["rl"]["lambda_"],
    )

    # 先 warmup 一些 episode
    for _ in range(10):
        ep = collector.collect_episode(policy, max_steps=300)
        replay_buffer.add_episode(ep)

    # 主循环
    total_steps = cfg["train"]["total_steps"]
    batch_size = cfg["train"]["batch_size"]

    for step in range(total_steps):
        # 继续收集真实数据
        ep = collector.collect_episode(policy, max_steps=300)
        replay_buffer.add_episode(ep)

        if replay_buffer.can_sample(batch_size):
            metrics = trainer.train_step(batch_size)

            if step % 10 == 0:
                print(
                    f"[{step:05d}] "
                    f"wm={metrics['wm_loss']:.4f} "
                    f"obs={metrics['obs_loss']:.4f} "
                    f"rew={metrics['reward_loss']:.4f} "
                    f"kl={metrics['kl_loss']:.4f} "
                    f"v={metrics['value_loss']:.4f} "
                    f"a={metrics['actor_loss']:.4f}"
                )


            if step % 50 == 0:
                avg_return = evaluate_actor(
                    env=env,
                    rssm=rssm,
                    actor=actor,
                    num_episodes=3,
                    max_steps=300,
                    device=device,
                )
                print(f"[Eval {step:05d}] avg_return={avg_return:.2f}")
                metrics_to_log["eval_return"] = avg_return


            metrics_to_log = {
                "step": step,
                **metrics,
            }
            logger.log(metrics_to_log)

            if step % 100 == 0:
                save_checkpoint(
                    path=f"checkpoints/step_{step:05d}.pt",
                    rssm=rssm,
                    obs_head=obs_head,
                    reward_head=reward_head,
                    actor=actor,
                    value_net=value_net,
                    wm_optimizer=wm_optimizer,
                    actor_optimizer=actor_optimizer,
                    value_optimizer=value_optimizer,
                    step=step,
                )




if __name__ == "__main__":
    main()