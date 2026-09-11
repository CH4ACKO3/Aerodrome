# Gymnasium World 适配器

`aerodrome.adapters.gymnasium.WorldEnv` 是可选模块。它包装已有原生 JAX World，不依赖 SB3 或任何训练框架，也不修改 World、计算图或物理模型。一个环境代表一个智能体控制的任务；World 内可以有多个实体，动作映射负责生成全部实体的输入。这不是多智能体 API。

```sh
uv sync --locked --extra gym
uv run --locked --extra gym python examples/gym_world.py
```

本轮安装并验证 Gymnasium 1.3.0。项目 WSL 脚本的 sync/test/check 已启用该 extra；直接使用其他 `uv run` 命令时需显式保留需要的 extras。

## 构建与使用

```python
from aerodrome.adapters.gymnasium import WorldEnv
from aerodrome.runners.episodes import Task

env = WorldEnv(
    world,
    initial_conditions=initial_by_entity,
    parameters=world_parameters,
    task=Task(observe, evaluate, max_episode_steps=500),
    task_parameters=task_parameters,
    action_space=action_space,
    observation_space=observation_space,
    action_to_inputs=action_to_inputs,
    jit=True,
)
try:
    observation, info = env.reset(seed=42)
    while True:
        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()
```

参数职责：

| 对象 | 接口与职责 |
|---|---|
| World | 实体、物理积分、tick/step；一个 Gym step 推进一个 World step |
| Task.observe | `(world_state, task_parameters) -> observation`，定义智能体允许看到的数据 |
| Task.evaluate | `(observation, public_action, next_observation, task_parameters) -> Outcome(reward, terminated)` |
| action_to_inputs | `(public_action, world_state, world_parameters) -> packed_world_inputs`，默认原样传入 |
| reset_factory | 可选 `(numpy_rng, options) -> ResetSpec(initial_conditions, world_parameters, task_parameters)` |
| spaces | 数值 Box、Discrete、MultiDiscrete、MultiBinary，或其 Dict/Tuple 组合 |

观测、奖励和结束逻辑复用现有 `runners.episodes.Task`。注意既有 EpisodeRunner 的 action 直接是 World inputs；如果 Gym 暴露归一化动作，就需要在原生训练路径也使用同一个 action_to_inputs，确保奖励看到相同的公开动作表示。WorldEnv 不会推断或改写这一语义。

`initial_conditions` 使用实体 ID 字典，`parameters` 使用 WorldParameters。配置系统的 `build_experiment` 可以构造 World 和参数，再交给该适配器；Gym 任务与空间由 Python 工厂提供，本轮未新增独立 Hydra runner 配置。

## reset 与 episode 语义

`reset(seed=...)` 调用 Gymnasium 的标准随机数初始化，再从该随机源生成 World seed；相同 seed、参数和动作序列可复现。`reset(seed=None)` 继续当前随机流。需要可复现的 `action_space.sample()` 时，另外调用 `env.action_space.seed(...)`。

reset_factory 只在宿主侧执行，适合域随机化、初值抽样和任务目标设置。无工厂时非空 options 会报错，避免静默忽略配置。工厂返回的状态和参数应保持固定 PyTree 结构、shape 和 dtype，以复用 JIT 编译。

任务结束返回 `terminated=True`；达到 Task.max_episode_steps 时返回 `truncated=True`。与既有 EpisodeRunner 一致，同一步已经 terminated 时不再标记 truncated。结束条件仅在 World step 边界检查，不在每个物理 tick 检查。返回的是结束时的观测，没有内部自动 reset；完成后继续 step 会抛出 ResetNeeded。

info 提供 `tick`、`time_s`、`episode_steps`、`episode_return`，不会默认暴露全部物理状态。动作检查失败不会推进仿真，也不会自动裁剪动作；transition 或输出校验失败后要求 reset。观测违反空间、观测/奖励非有限时明确报错，不伪造成功或奖励。隐藏状态的物理有效性应由模型和任务自身处理。

## JAX 与向量化边界

WorldEnv.step 在宿主侧返回 NumPy 观测、Python 标量 reward/bool，以及 dict info。每个决策步会发生设备到宿主的同步；内部 World.step + 动作映射 + observe/evaluate 可以融合 JIT，并丢弃未使用的完整 tick 轨迹。

同一适配器还暴露纯数组函数：

```python
next_state, next_observation, reward, terminated = env.transition(
    world_state, observation, action, world_parameters, task_parameters
)
```

它可组合 `jax.jit/vmap/grad/scan`，不含 Gym 或 NumPy 操作，也不包含宿主 episode 计数和自动重置。原生 GPU 大批量训练使用该函数或已有 BatchedWorld/EpisodeRunner；Gymnasium 的 SyncVectorEnv 可以包装多个 WorldEnv，但不是 GPU 原生 vmap。未在本轮验证 AsyncVectorEnv 多进程执行或第三方训练库。

目前支持 `render_mode=None` 和文本 `ansi`，close 幂等。该适配器不拥有外部渲染器、设备或 MATLAB 进程；在线图形与 host 联合仿真仍使用已有专用模块，没有通过此接口承诺支持这些会话。

可运行例子 `examples/gym_world.py` 用三轴力控制刚体速度，展示 seed、初值 options、奖励、终止条件和简单比例策略，且运行官方 `check_env`。接口约定参见 [Gymnasium Env](https://gymnasium.farama.org/api/env/)。
