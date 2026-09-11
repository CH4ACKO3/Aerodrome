# 按依赖推进：tick 内与跨 tick 的异步计算

状态：已实现独立原生实体的任务图执行器，并提供通用任务 DAG 接口。

后续扩展已落地：[模块端口图编译器](module-compiler.md) 可从显式端口连线生成事件依赖与 JAX 区域；本页介绍它使用的底层任务执行器。

## 1. 逻辑时间与执行时间分离

原先的 `WorldState.tick` 表示一个所有实体对齐的快照。现在增加宿主执行器，运行中各实体可以处于不同逻辑时刻，最终返回对齐的 WorldState。
不会因为 CPU/GPU 完成顺序不同而改变传感器时间、控制频率或随机序列。

```text
A@0 → A@1 → A@2 → A@3 ─┐
                        ├→ 统一记录/返回
B@0 → B@1 → B@2 → B@3 ─┘
```

A@1 不依赖 B@0，因此 A 可以先推进。只有真正依赖的结果尚未产生时才等待。
这与函数返回是否使用 `async def` 是两个问题：当前 `run()` 等待完整结果，但内部工作线程按图独立推进。

## 2. World 使用方式

```python
from aerodrome.runners.graph_world import GraphWorldRunner

runner = GraphWorldRunner(world, max_workers=2, chunk_ticks=32)
final_state, trace = runner.run(
    state, inputs, parameters, steps=500,
    on_complete=lambda task_id, output: print("完成", task_id),
)
```

前提与 `world.rollout` 一样：外部 inputs 在整段 rollout 保持，数值 parameters/resources 不在执行中修改。
不能把这种超前推进用于尚不知道下一次策略输入的在线交互；策略产生动作的任务必须纳入依赖图，或在决策边界结束本次 run。

`chunk_ticks=32` 把一个实体连续 32 个 tick 编译成一次 JAX scan。
调度器只处理这些较大的任务；最后一个任务可短于 32 tick，也允许 chunk 跨决策边界。
同一个实体的任务串联，不同实体的任务链独立。现有 World 不支持实体间交互，所以可以这样直接拆分。

`plan()` 返回可检查的 `WorldGraphPlan`：每个 task ID 为 `(entity_id, tick_offset)`，offset 相对传入状态的时刻。
`run()` 返回与原 `world.rollout` 相同的状态与 `[decision_step, physics_tick, ...]` 记录。
这是宿主入口，不能整体包进 `jax.jit/grad/vmap`。端到端自动微分继续使用原生 World 的纯函数路径。

运行示例：

```shell
python examples/graph_world.py
```

它推进两个独立实体，与原整体 rollout 比较，并写入 `artifacts/graph_world/summary.json`。
完成顺序不是固定输出；轨迹和逻辑时间必须固定。

## 3. tick 内部的计算图

`TaskGraph` + `DependencyExecutor` 支持任意显式 DAG。例如给定同一阶段的状态和舵偏后，互不依赖的气动和发动机输出可以先分支、再汇合：

```python
from aerodrome.runners.dataflow import Task, TaskGraph, DependencyExecutor

graph = TaskGraph((
    Task("aero@k", lambda _: compute_aero(stage_state)),
    Task("engine@k", lambda _: compute_engine(stage_state)),
    Task("forces@k", lambda d: combine(d["aero@k"], d["engine@k"]),
         ("aero@k", "engine@k")),
))
outputs = DependencyExecutor(max_workers=2).run(graph)
```

以上是接口示意，领域函数由具体模型提供。调度器验证依赖存在、ID 唯一和图无环；任务只能通过依赖映射读取声明的上游结果，并且应将结果视为不可变。
Python 闭包的隐式读取/副作用不可能靠这些检查自动识别。真实仿真必须显式描述状态版本和所有因果依赖；节点名称本身不构成时间正确性证明。

当前 `GraphWorldRunner` 将整个实体的一个 chunk 当作任务，**不会自动拆开 PitchAssembly 内部的 sensor/KF/PID**。
要在内部获得宿主级并发，可以显式提供上述 TaskGraph，或使用新增 ModuleGraph 自动生成计划。`SystemSpec` 仍是校验元数据，不是可直接编译成任意计算图的程序。

对于轻量 JAX 气动函数，通常先保留在同一个 JIT 数值块中，由编译器处理算子调度。
强耦合连续系统的每个积分阶段仍需重算联合 RHS；不能为了并行，把 RK4 改成只计算一次气动力。

## 4. 跨实体依赖怎样表达

若 A 的下一状态依赖 B 的当前状态，任务边应明确为 `B@k → A@(k+1)`，并保留 `A@k → A@(k+1)`。
若双方相互依赖当前状态，它们只能在已有数据允许的范围内超前；不会因为任务异步而自动消除物理依赖。
同一时刻的直接馈通环需要联合求解或有物理依据的延迟，不能靠执行顺序或添加虚假延迟解决。

已有通用 DAG 可以手工表达这些依赖；World 的自动跨实体连线/时间偏移编译器尚未实现。
节点依赖必须引用指定版本的信号，不允许读取“当前最新值”，否则机器负载会改变仿真模型。
共享只读气动表不会产生推进依赖；共享可变风场、碰撞、共同控制器、全局终止条件或在线策略可能产生依赖。

## 5. 执行机制与能力边界

- 调度器只提交依赖全部完成的节点，最多运行 max_workers 个任务；没有逐 tick 的全局屏障。
- JAX 返回数组时计算可能仍在设备执行。工作线程在结果 `block_until_ready` 后才报告任务完成，不复制到 NumPy。
- 线程调度器位于 JAX 变换之外。JAX 允许从不同宿主线程调用 API，但不允许在正在追踪的函数内并发操作 tracer。
- 观察到任务失败后停止提交新任务，失败后继不会执行，取消未启动工作，并等待已启动工作结束后抛出带节点 ID 的异常。
- 线程不能安全强制终止。没有超时中断、回滚和部分 WorldState 提交，也不支持多控制器 JAX collective 的跨进程调度。
- 首版展开有限任务图，并保留节点结果用于拼接轨迹；不是无限时长的流式执行器。较长仿真应选合理 chunk 和分段长度。

MATLAB 有状态会话需独占所有者，按它自己的时间链串行调用，还要处理适配器线程亲和性、超时与关闭。
本次没有将现有 `CoSimulationRunner` 改成并发执行器，也没有宣称 MATLAB 适配器已经可用。

同一 GPU 上两个可运行任务不保证同时占用计算单元。多个线程可能竞争设备，纯 Python CPU 任务也受解释器锁影响。
同构大批量原生环境可继续选择 `vmap/scan`；异构/慢速外部组件更可能从独立任务调度中获益。需要完成同步后的实际耗时测试才能判断。

依据：[JAX 并发约束](https://docs.jax.dev/en/latest/concurrency.html)、[JAX 异步派发与计时](https://docs.jax.dev/en/latest/async_dispatch.html)。

## 6. 验证

- 用线程 Event 握手证明：慢实体第 0 步未结束时，快实体第 1 步已经执行；没有依赖不可靠的耗时比或 sleep。
- 同一 tick 的两个分支并发启动，汇合节点只在两者结束后运行。
- chunk 为 1、5、64 tick，从非零逻辑时间开始，两实体的完整状态/日志与原生 rollout 一致。
- 验证图的未知依赖、重复 ID、环以及失败后继拒绝和在途任务排空。
