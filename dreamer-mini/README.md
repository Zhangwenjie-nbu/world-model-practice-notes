真实环境
   ↓
收集 (obs, action, reward, done)
   ↓
Replay Buffer
   ↓
采样序列 batch
   ↓
RSSM:
   上一步 latent + action → h_t
   h_t → prior(z_t)
   h_t + obs_t → posterior(z_t)
   ↓
feature = concat(h_t, z_t)
   ↓
heads:
   obs_head(feature)
   reward_head(feature)
   continue_head(feature)
   ↓
world model loss

posterior states
   ↓
imagination rollout (只用 prior + actor)
   ↓
imagined features
   ↓
reward head + critic
   ↓
lambda return
   ↓
actor loss / critic loss



envs/make_env.py

负责：

创建环境

统一 reset / step 接口

返回 obs_dim / action_dim 等信息


data/replay_buffer.py

负责：

存储 episode 数据

支持采样长度为
𝑇
T 的子序列

输出 batch tensor


models/rssm.py

负责：

recurrent transition

prior step

posterior step

rollout observe / imagine

这是世界模型最核心的文件。


models/heads.py

负责：

observation head

reward head

continue head


models/actor.py

负责：

根据 latent feature 输出动作分布


models/critic.py

负责：

根据 latent feature 输出 value



training/losses.py

负责：

world model loss

actor loss

critic loss

lambda return



training/imagination.py

负责：

imagination rollout

latent trajectory 生成


training/collector.py

负责：

用当前 actor 去环境中采样

初期也可以混合 random exploration


training/trainer.py

负责：

串联整个训练步骤

管理 optimizer

调用各模块



train.py

负责：

读取配置

创建所有模块

启动训练主循环