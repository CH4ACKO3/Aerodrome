# 多 World 与批量 rollout

`BatchedWorld` 对同一个静态 `World` 拓扑使用 `vmap`，时间推进使用 `lax.scan`；外层 `jax.jit` 可以编译整个 rollout。没有为每个 World 创建 Python 线程，也不需要手写 Triton。全 JAX 的 `GraphAssembly` 可以直接复用。带 MATLAB/宿主模块的图仍需外部协调器，不在这个纯数值 batch 中执行。

可运行入口：`uv run --locked python examples/batch_rollout.py`。Windows 使用 `./scripts/run-wsl.ps1 examples`，结果在 `artifacts/batch_rollout/`。

## 数据轴与共享参数

| 数据 | 轴 |
|---|---|
| WorldState 的数值叶子 | `[B, ...]` |
| `step` 的完整日志 | `[B, S, ...]` |
| `rollout` 的完整日志 | `[T, B, S, ...]` |
| 自定义单步投影、训练 Transition | `[T, B, ...]` |
| 批量动作序列 | `[T, B, ...]` |

`T` 是决策步数，`S` 是每个决策步的物理 tick 数。World 内部的状态、模型公式和单实体接口不增加 batch 维；这个维度由外层 `vmap` 引入。

```python
from aerodrome.runners.batch import BatchedWorld

batch = BatchedWorld(world)  # 默认参数和初始条件共享，输入按 axis=0 映射
state = batch.reset(seed=42, world_ids=[100, 101], initial_conditions=initial)
batch.validate(state, batched_inputs, parameters)
run = jax.jit(lambda s, u, p: batch.rollout(s, u, p, steps=256))
final, trace = run(state, input_sequences, parameters)
```

`parameter_axes`、`initial_axes` 和 `input_axes` 接受 JAX `in_axes` 的 PyTree 前缀：`None` 表示共享，`0` 表示该字段第一维是 World。不要为共享气动表显式复制 B 份。

例如 `WorldParameters((0,), None)` 表示一个实体的所有参数都批量化，`resources` 共享。若仅阻尼需要 domain randomization，可将其他参数叶子的轴设为 `None`，阻尼轴设为 `0`；示例展示了这类混合参数。导航器仍使用共享的标称模型，不会自动读到真实的随机机体参数。当前数值参数在一段 rollout 和自动重置之间保持不变；若要每个 episode 重采样机体参数，需要再扩展显式参数状态，不能仅依赖初始状态采样器。

## 初始条件、随机流与重置

`initial_sampler(key, template)` 是可选纯函数，用来按 episode 生成初始条件。默认直接返回模板。采样器必须可以 JAX trace，并保持 PyTree 结构、shape 和 dtype。导航初始先验应单独配置，不能无意中从真值复制。

随机键依次按 `seed → world_id → episode_id → stream` 派生；模型内部再按实体 ID 派生。初始化采样、模型和策略使用不同 stream。策略每步随机键还包含 episode 内的决策步数。因此：

- 同一稳定 ID 在 batch 重排或拆分后保留随机流；重排时也要同步重排逐 World 参数和初始模板。
- `reset_where(state, bool_mask, initial)` 只更新选中的状态和 episode ID；其余 World 的状态、时钟、模型随机键保持不变。
- 自动重置后本地 World tick 从零开始，适合 episode 内多速率调度。跨 episode 的实验时间应由调用方单独记录。
- `world_id` 和 `episode_id` 为 uint32，实验应避免 episode 计数溢出。`reset` 在宿主侧检查 ID 唯一性和取值；`initialize` 是供编译路径使用的入口，要求调用方提供合法 ID 数组。

固定 batch 的重置采用候选状态加 mask 选择，有重置时可能为全部槽位计算候选初始状态。这保证输出语义独立，不承诺只对选中槽位消耗计算。当前没有死亡槽位冻结、动态压缩或 padding/active mask 接口。

## 训练与评测的 episode 层

`EpisodeRunner` 将 episode 生命周期放在物理 World 之外：

```python
task = Task(
    observe=observe,        # (single_world_state, task_params) -> observation
    evaluate=evaluate,      # (obs, action, next_obs, task_params) -> Outcome(reward, terminated)
    max_episode_steps=500,
    parameter_axes=0,       # 此例每个 World 有自己的任务参数
)
runner = EpisodeRunner(batch, task)
env = runner.initialize(state, task_parameters)
```

`observe` 是可审查的观测边界：它可以选取估计值、协方差、任务目标等。策略只收到该观测、策略参数和随机键，不收到 WorldState 或真实机体参数。这个边界依赖任务作者正确编写 `observe`，不是对任意 Python 代码的安全隔离。

每步先推进完整的决策步，然后判断终止。`terminated` 是任务条件，`truncated` 是达到步数上限；若同一步满足两者，终止优先。完成的槽位立即自动重置，其他槽位继续。不会在物理子步中间提前停止。

`Transition.next_observation` 始终保留重置前的末端观测，包括 terminal observation；返回的 `EpisodeState.observation` 则是下一次策略应该读取的观测，完成槽位已经换成新 episode 的初始观测。训练代码应按算法处理终止和时间截断的 bootstrap，不能把重置后的观测当成旧 episode 的下一状态。

日志还包含旧 episode 的 `world_id`、`episode_id`、从 1 开始的 `episode_step`、累计 `episode_return`、动作和奖励。奖励 dtype 默认 float32，可通过 `Task.reward_dtype` 指定。传入的动作必须全部具有 batch 轴；终止条件和奖励分别是每 World 的标量 bool 与浮点数。

## 策略与跨 chunk 续算

```python
policy = Policy(
    initialize=initialize_policy,  # (obs, policy_params, key) -> recurrent_state 或 ()
    step=policy_step,              # (memory, obs, policy_params, key) -> (memory, action)
)
carry = runner.start_policy(env, policy_parameters, policy=policy)
run = jax.jit(lambda c, weights: runner.rollout_policy(
    c, weights, parameters, initial, task_parameters, policy=policy, steps=256))
carry, transitions = run(carry, policy_parameters)
carry, next_transitions = run(carry, policy_parameters)
```

策略的循环状态会与对应 episode 一起重置；无状态策略返回 `()`。传回完整 carry 即可跨 chunk 延续策略、物理状态和随机序列，不能重新调用 reset。固定动作序列则使用 `runner.rollout`；它不会根据重置而重新播放输入序列，输入仍按全局扫描步索引。

全部回调必须是纯 JAX 数值函数。固定拓扑、batch 大小、steps、叶子 shape/dtype 共同影响编译缓存；改变它们可能重新编译。连续物理路径支持自动微分，离散终止、重置和阈值分支不保证平滑梯度。

## 记录与设备

`BatchedWorld.rollout` / `rollout_constant` 的 `record` 有三种模式：`"full"` 保存所有物理日志；`None` 返回空记录；函数 `(next_single_world_state, single_step_trace) -> projection` 只保留需要的量。投影在编译路径内执行，使编译器有机会消除无用日志，不应在函数内写文件。

`EpisodeRunner` 默认保存紧凑的训练 Transition，不保留完整物理日志；需要真值诊断时另跑固定输入评测或增加显式评测记录接口。

当前已验证 Python 3.14t / JAX CPU。数组和计算采用 JAX 的设备放置，未来安装合适 GPU 运行时并将输入放到设备后可复用该结构，但本次未实测 GPU、吞吐、显存占用或多卡 sharding。计时应先完成预热，并对输出调用 `block_until_ready()`，避免把异步提交时间当作运行时间。
