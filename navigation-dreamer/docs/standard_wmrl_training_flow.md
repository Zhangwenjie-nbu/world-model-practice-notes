# Standard WM-RL 训练流程与问题修复记录

本文档说明当前 `scripts/train_agent_standard_wmrl.py` 这一条训练链的最终流程、训练中反复出现的塌缩问题、问题的根因，以及最后是如何修到可稳定收敛的。

当前这套实现的目标不是“最纯论文版 Dreamer 复现”，而是：

- 保留标准 RSSM world model 主干
- 不显式输入 `prev_action` 给 actor
- 不依赖外部专家标签做主训练
- 让视觉导航避障任务真正学起来，并且后期不容易漂移塌缩

参考代码：

- [scripts/train_agent_standard_wmrl.py](/home/wenjiezhang/RL/world_model_practice_notes/navigation-dreamer/scripts/train_agent_standard_wmrl.py)
- [trainers/joint_trainer_standard.py](/home/wenjiezhang/RL/world_model_practice_notes/navigation-dreamer/trainers/joint_trainer_standard.py)
- [memory/replay_buffer.py](/home/wenjiezhang/RL/world_model_practice_notes/navigation-dreamer/memory/replay_buffer.py)
- [envs/grid_nav_env.py](/home/wenjiezhang/RL/world_model_practice_notes/navigation-dreamer/envs/grid_nav_env.py)

## 1. 当前最终训练流程

### 1.1 环境与动作

环境是 2D 视觉导航环境，动作空间只有三个：

- `forward`
- `turn_left`
- `turn_right`

当前标准训练默认使用：

- 碰撞立即终止：`terminate_on_collision=True`
- `goal_reward=20.0`
- `collision_penalty=8.0`
- `turn_penalty=0.05`
- `progress_reward_scale=1.0`
- `step_penalty=0.1`

设计意图：

- 碰撞是真正严重失败，应该在真实环境里及时终止
- 转向必须有成本，但不能贵到策略宁可撞也不愿转
- 路径缩短奖励要足够清晰，让正确转向真正有价值

### 1.2 Replay 结构

训练数据全部来自真实环境 rollout，存入 episode replay buffer。

buffer 里同时维护两类数据：

- regular episodes
- success episodes

success episodes 单独保留的原因不是为了“专家训练”，而是：

- world model 需要更多稀有成功终止样本
- actor 后期需要一个来自真实成功轨迹的行为锚点

当前 replay 支持：

- 常规 batch 采样：`sample_batch()`
- 成功轨迹 batch 采样：`sample_success_batch()`

### 1.3 World Model 主干

当前标准版 world model 包含：

- `encoder`
- `rssm`
- `decoder`
- `reward_model`
- `continue_model`
- `event_model`

其中：

- `decoder` 负责重建图像
- `reward_model` 预测 reward
- `continue_model` 预测 episode 是否继续
- `event_model` 预测两个显式事件：
  - success
  - collision

这部分的核心训练目标在 [trainers/joint_trainer_standard.py](/home/wenjiezhang/RL/world_model_practice_notes/navigation-dreamer/trainers/joint_trainer_standard.py) 里：

- reconstruction loss
- reward loss
- continue loss
- event loss
- KL balancing loss

关键点：

- posterior feature 和 prior feature 都参与监督
- success / terminal reward 在 reward loss 里加权
- event head 对稀有 success / collision 提供显式监督

### 1.4 Actor 输入

当前 actor 不直接使用 `rssm.get_feat(state) = concat(deter, stoch)`。

而是统一使用：

- `actor_feat = concat(deter, mean)`

原因：

- `stoch` 是采样值，会引入额外噪声
- actor 对输入噪声非常敏感，更容易抖动和塌缩
- `mean` 更平滑，适合做策略输入

这里最重要的不是“哪种更标准”，而是：

- actor 训练时和 actor 推理时必须保持一致

当前 real rollout、evaluation、BC、AC imagination 都已经统一到 `deter + mean`。

### 1.5 Actor-Critic 训练

当前 AC 路径仍是标准的 latent imagination 风格：

1. 从真实 replay 的 posterior state 里采样 imagination 起点
2. 用 actor 在 latent space 里 rollout
3. 用 reward / continue / critic 计算 imagined return
4. 更新 actor 与 critic

但这里已经做了几项关键工程改造：

- actor imagination 起点会排除 collision 状态
- imagined reward 里显式加入：
  - success bonus
  - collision penalty
- collision 风险会下调 effective continue
- critic 同时看 imagined TD 和 real-data TD

这意味着 actor 不是只靠一个模糊的 reward regression 学避碰，而是能直接从显式 collision 风险得到负反馈。

### 1.6 成功轨迹 BC 锚定

当前 actor 不再是“纯 imagined AC”。

每次 AC 更新后，还会额外做轻量的 success-replay BC：

- 数据来源不是外部专家
- 数据来源是 replay 中真实成功过的 episodes
- 目标是防止 actor 后期漂移到单一动作模式

这一部分非常关键。

它的本质不是“回到 supervised 学习”，而是：

- 用真实成功动作分布给 actor 一个稳定锚点
- 防止 imagined AC 在模型偏差下把策略慢慢推坏

### 1.7 Actor 相对 Critic / World Model 的步幅控制

这是最后稳定下来的一个关键修改。

当前实现里 actor 明显比 critic / world model 更保守：

- actor 学习率更低：`1e-6`
- actor AC 更新不是每步都做，而是由 `actor_ac_interval` 控制，默认每 4 步做一次
- world model 和 critic 保持更高的更新频率

这样做的直观含义是：

- 先让模型和价值函数跟上
- 再慢慢推动 actor
- 避免 actor 过快利用 world model 偏差

## 2. 当前训练节奏

当前主训练节奏可以概括为：

1. 初始随机收集 replay
2. 训练 world model
3. actor 启动后：
   - 低频 imagined AC
   - 高频 world model / critic 更新
   - success replay BC 锚定
4. 周期性 evaluation
5. curriculum 从短路径逐步放宽到 full random

这套节奏的核心思想是：

- world model 先稳定
- actor 再慢慢动
- actor 每次动完，都用真实成功轨迹做轻微拉回

## 3. 训练中遇到的主要问题

下面按实际出现的顺序说明问题。

### 3.1 Actor 输入分布错位

最早的问题之一是：

- BC 时 actor 用 `deter + mean`
- AC rollout / eval 时 actor 用 `deter + stoch`

结果：

- 训练分布和部署分布不一致
- actor 很容易在测试时抖动、退化、塌缩

修复：

- actor 所有路径统一使用 `deter + mean`

### 3.2 纯 reward / continue 监督不够，actor 会塌到高碰撞策略

早期标准版为了“更纯”，拿掉了显式 event head。

结果是：

- actor 只能通过 reward 回归间接学 collision 风险
- 在导航任务里，collision 是稀有但关键事件
- world model 一旦对 collision 估计偏乐观，actor 就会学出“直冲碰撞”的策略

现象包括：

- 几乎全 `forward`
- 高 collision rate
- imagined continue 虚高

修复：

- 把 `event_model` 接回标准版
- 显式监督 `success` 和 `collision`
- 在 actor imagined return 里直接加入 collision penalty
- collision risk 下调 effective continue

### 3.3 人工动作正则可能把策略推向另一种塌缩

中间为了解决转圈问题，尝试过一些动作层面的手工惩罚，例如：

- `action_repeat_penalty`
- 非 `forward` 惩罚

这类正则的风险是：

- 太强会把 actor 推向单边策略
- 例如全 `turn_left`
- 或者反过来推成全 `forward`

其中 `action_repeat_penalty` 还有一个额外问题：

- 它依赖动作历史
- 不是环境原始 reward 的 Markov 项

最终处理原则是：

- 删除会明显破坏目标结构的历史型动作惩罚
- 只保留少量、可控的、与真实任务一致的信号
- 更重要的是修正 world model 风险建模和 actor 步幅，而不是依赖动作黑名单

### 3.4 Success-replay BC 一开始接错了监督对齐

这是后面发现的一个很关键的 bug。

最初 success-replay BC 用的是：

- `state_t -> actions_t`

但在本项目的数据对齐里：

- `actions[:, t]` 是“到达 `obs[:, t]` 前执行的动作”

所以 actor 在当前状态下真正应该学的是：

- `state_t -> action_{t+1}`

错误对齐会导致：

- BC 看起来精度很高
- 但本质上在教 actor “你是怎么来到这里的”
- 而不是“你现在该怎么走”
- 结果 actor 容易被错误 BC 稳定推向 `forward`

修复：

- success BC 改成 1-step shift
- 用 `posterior[:, :-1]` 对齐 `actions[:, 1:]`
- `bc_acc` 和 `bc_act` 也按这个新对齐统计

这是一个结构性 bug，修复后效果提升非常明显。

### 3.5 后期策略漂移

即使中期已经学得很好，后期仍然可能慢慢漂坏。

典型现象：

- 中期 deterministic eval 很高
- 继续训练后 success 掉下去
- collision 回升
- `critic_imag` 和 `critic_real` 开始分叉

根因不是“突然不会了”，而是：

- actor 的优化目标主要来自 imagined rollout
- 一旦 world model / critic 在某些区域有系统偏差
- actor 会继续利用这种偏差，把已经有效的策略推离稳定点

这本质上是：

- actor 相对 critic / world model 更新过快
- actor 过度利用模型偏差

修复方向不是继续堆 reward，而是：

- 降低 actor 步幅
- 提高 critic / world model 相对稳定性
- 保留 success replay BC 作为锚点

最终稳定下来的关键改动就是：

- actor learning rate 从 `3e-6` 降到 `1e-6`
- actor AC update 频率降低到每 4 步一次

## 4. 为什么“降低 actor 相对 critic/world model 的步幅”有效

这是当前版本能最终稳定收敛的核心原因之一。

直觉上可以这样理解：

- world model 和 critic 在不断修正对环境的近似
- actor 是最容易利用这些近似误差的模块
- 如果 actor 每步都动，而且动得太快
- 那么它总能比模型修正更快地朝“模型漏洞”方向漂移

把 actor 放慢以后：

- critic 有更多真实数据 TD 机会校正价值
- world model 有更多机会修正 reward / continue / collision 预测
- actor 每次更新面对的是更稳定的价值面

因此：

- 不容易后期从好策略漂到坏策略
- deterministic policy 更稳定
- stochastic policy 也更容易跟着提升

## 5. 当前版本的关键稳定器

从最终可收敛结果来看，当前版本最重要的稳定器有以下几个。

### 5.1 显式 event head

作用：

- 让 success / collision 这种稀有事件不再只靠 reward 回归间接学习

### 5.2 统一 actor feature

作用：

- 避免 actor 训练 / 推理输入错位
- 减少 sampled latent 噪声导致的抖动

### 5.3 Success replay BC 锚定

作用：

- 给 actor 持续提供“真实成功动作分布”的约束
- 防止 imagined AC 把策略洗成单一动作

### 5.4 正确的 BC 目标对齐

作用：

- 教 actor 学“当前状态的下一动作”
- 而不是“到达当前状态的上一动作”

### 5.5 更慢的 actor 更新

作用：

- 限制策略漂移
- 减少 actor 对模型偏差的放大

## 6. 当前结果说明

当前一条成功训练 run 在 `step=100000` 时达到：

- deterministic eval on full random:
  - `avg_reward=28.344`
  - `success=0.990`
  - `collision=0.000`
  - `avg_length=15.7`
  - `avg_probs=[0.54, 0.226, 0.234]`
- stochastic eval on full random:
  - `avg_reward=23.591`
  - `success=0.910`
  - `collision=0.090`
  - `avg_length=29.0`
  - `avg_probs=[0.44, 0.279, 0.281]`

训练日志同时显示：

- world model loss 已经很低
- imagined action 分布没有塌成单动作
- success replay BC 仍然在提供有效锚定
- critic imagined / real 差距处于可接受范围

这说明当前版本已经从“能不能学起来”，进入“如何更稳、更泛化、更可解释”的阶段。

## 7. 当前版本的经验总结

如果只保留一句话，那么当前这套标准 WM-RL 能稳定训练起来，主要不是靠某一个 reward trick，而是靠下面这组组合：

- world model 显式学 success / collision
- actor 全路径统一使用 `deter + mean`
- success replay 给 actor 一个真实成功行为锚点
- BC 监督必须严格按时间对齐
- actor 必须比 critic / world model 更新得更保守

这几条里，最后一条最容易被低估。

在视觉导航这类任务里，纯 imagined AC 很容易中后期策略漂移。当前最终版本真正解决问题的关键，不是再加更多 reward shaping，而是：

- 先把监督对齐和风险建模修对
- 再控制 actor 的更新速度

这也是当前脚本最终能够在 full random 上稳定达到高成功率、低碰撞率的主要原因。
