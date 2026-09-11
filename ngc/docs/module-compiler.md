# 模块端口图与跨 tick 编译

本次新增可执行 `ModuleGraph`：显式端口连线、当前/上一 tick 信号版本、更新周期、JAX/宿主后端声明，编译成一个或多个可运行子图。仍保留具体模型的独立方程函数。

入口：`examples/compiled_modules.py`。它是一维速度跟踪演示，不是 F16：理想速度观测、比例油门、每 4 tick 更新的静态推力映射、Euler 速度积分。

## 1. 最小模块契约

```python
Module(
    id="control",
    inputs=(estimate_port, target_port),
    outputs=(throttle_port,),
    step=control_step,
    backend="jax",  # 或 "host"
    every=1,
    equation="throttle = clip(K*(target-estimate), 0, 1)",
)
```

其中 `control_step(state, inputs, parameters, context)` 返回 `ModuleValue(next_state, outputs)`。
state 为固定结构的数值 PyTree，无状态模块用 `()`；outputs 为端口名到数组的字典。
context 提供 tick、time_s、physics_dt_s、sample_period_s。多速率更新在绝对 tick 可整除 every 时触发，其他时刻保持内部状态和最后输出。

每个模块只拥有自己的状态。JAX 模块必须使用可追踪的纯数值代码；随机键也必须放入显式状态，不能在函数中隐式修改全局 RNG。
这不是把所有物理模型都变成独立积分器：强耦合连续状态仍应由同一个物理模块调用联合 RHS 和积分器。

## 2. 连线与时间语义

```python
Wire(("observation", "estimate"), ("control", "estimate"), delay=0)
Wire(("body", "next_speed"), ("observation", "speed"), delay=1)
InputBinding("target", ("control", "target"))
```

- `delay=0` 读取源模块在当前 tick 的输出；源模块本 tick 未更新时读取明确保持的上次输出。
- `delay=1` 读取源模块截至上一 tick 的输出；运行段开始时由初始 ModuleValue.outputs 提供历史保持值。
- 首版只支持 0/1 tick 延迟，未提供任意延迟队列、乱序消息或插值。
- 端口校验包括名称、shape、dtype、单位、坐标系、物理量和参考点。转换必须通过显式模块。
- 每个输入必须恰好有一个来源；当前 tick 的图必须无环。状态反馈通过有依据的时间边表示，不会自动添加延迟。

**tick 是事件索引，字段对应的物理时刻仍由模块公式定义。** 示例 body 在事件 k 积分得到 `next_speed=v_(k+1)`，下一次观测通过 delay=1 读取它，因此控制器使用 v_k。日志中的 body.next_speed 位于区间终点；observation.estimate 位于起点。命名与公式不能省略这个区别。
初始 body.state 和 body.outputs.next_speed 均为 v_0，用来明确提供第一个事件需要的状态和历史信号。框架不会从真值自动生成导航先验。

## 3. 编译与执行

```python
program = CompiledModuleGraph(
    graph, ticks=128, physics_dt_s=0.01,
    chunk_ticks=32, start_tick=0,
)
final, trace = program.run(initial, input_sequences, parameters)
plan = program.describe()
```

initial 是 `ModuleGraphState(next_tick, {module_id: ModuleValue(...)})`。
parameters 按模块 ID 显式提供；input_sequences 按外部端口名提供 `[ticks, *port.shape]` 数组。
因此编译块可以使用完整预知的输入序列，不要求整块恒定。在线外部输入应建成宿主模块，或在输入尚未知处结束本段。

输出 trace 包含 tick、各模块发布输出和 updated 掩码。每个物理 tick 都有一条记录，低频模块保留采样保持值。
分段运行需要用上段 final 作为下段 initial，并按对应 start_tick 构建计划。段开始之前最后一次发布的输出由 initial 保存，不能重置成零。

## 4. 自动分区算法

执行顺序为：

```text
模块 + 端口 + 延迟
       ↓ 端口检查、当前 tick 拓扑排序
有限时间事件 DAG：(module_id, tick)
       ↓ 追踪状态前驱、信号生产者和宿主祖先
依赖安全的 JAX 区域 + 宿主事件
       ↓ 区域生成组合函数、jax.jit
DependencyExecutor 就绪调度
```

一个事件依赖本模块上一次更新的状态，以及每个输入所引用的确切生产事件。
未触发更新的低频模块不生成执行事件，连线直接引用它上一次发布的结果。

融合条件是两个相连 JAX 事件处于同一个 chunk 时间窗口，且具有完全相同的宿主祖先集合。
这个条件比较保守，但能防止把 `JAX A → host H → JAX B` 错合成一个需要先输出再等待输入的不可执行区域。
合并后再次验证区域图无环。没有连线的独立实体/模块仍保留独立区域。

例如 engine 每 4 tick 在宿主更新一次，后续 JAX 观测/控制/机体在不需要下一次 engine 输出的范围内可以跨 tick 合并。闭环中，为下一次宿主调用准备输入的节点也可能包含在前一个区域，所以区域边界按依赖决定，不只是按统一 tick 切割。
改变 every 会改变采样保持模型，编译器不会为了减少通信而擅自改变它。

## 5. 两种 JAX 编译路径

**分区路径 `program.run`：** 每个 JAX 区域生成一个组合函数并 JIT；区域内的有限事件图在追踪时展开。Python 调度只发生在区域边界。chunk_ticks 限制展开的时间窗口，避免无界增长。

**全原生路径 `program.native`：** 所有模块为 JAX 时提供 `jit(lax.scan(...))`，单个可执行程序包含整个多 tick 循环，支持 JAX 梯度和批变换。模块图拓扑在单个 tick 内静态展开，时间循环保留为 scan。存在 host 模块时该入口为 None。

```python
compiled = program.native.lower(initial, input_sequences, parameters).compile()
final, trace = compiled(initial, input_sequences, parameters)
```

program.native 使用传入 state.tick 推进绝对时钟；program.run 的宿主事件计划要求 tick 与 start_tick 一致。
数值参数与输入序列是函数参数，不作为静态编译常量；shape/dtype 改变可能触发重新编译。
program.run 会做宿主输入检查；直接调用 native 属于纯数值入口，调用者需提供契约正确的输入。

一个 JAX 编译程序可以包含循环和多个设备算子，不承诺只有一个 CUDA kernel。
宿主任务图不会被放进 JIT 或自动微分；完整可微闭环使用 native。

## 6. 接回现有 World

```python
assembly = GraphAssembly(graph)
world = build_world(WorldSpec((EntitySpec("aircraft", assembly),)))
state = world.reset(seed, {"aircraft": initial_module_values})
inputs = world.pack({"aircraft": inputs_for_one_tick})
parameters = world.parameters({"aircraft": module_parameters})
next_state, trace = world.step(state, inputs, parameters)
```

GraphAssembly 限定为全 JAX 图，复用 World 的调度、tick/step、scan、vmap 和独立实体 GraphWorldRunner。
初值可以是映射，也可以是接受实体 PRNG key 的初值工厂。World 共享 resources 不会自动注入 ModuleGraph，模型应显式通过参数声明所用数值资源。
含宿主模块的图通过 CompiledModuleGraph 执行，不能假装其会话状态可复制进 World 的 JAX PyTree。

原有 PitchAssembly 和 SystemSpec 仍可直接使用。SystemSpec 是检查/说明信息；需要自动生成连线与编译计划的模型，应使用新增的可执行 ModuleGraph。

## 7. 宿主与实现范围

backend="host" 的函数不会被 JAX 追踪，同一模块的更新由显式状态依赖保证串行。不同模块可由线程池并行执行。
模块之间传递值，依赖值视为不可变；不支持共享可变状态或靠闭包隐式传递信号。
当前示例宿主发动机只是进程内 NumPy 测试组件，没有调用 MATLAB。
真实 MATLAB/FMU 适配器仍需提供专属会话所有权、线程亲和性、时间检查、超时和关闭逻辑；不能让两个模块捕获同一会话并假定此编译器会自动发现。

首版保留所有事件结果用于完整轨迹与检查，并按有限 horizon 构建计划。尚未实现流式窗口回收、跨进程执行、最优分区/代价模型、自动识别 Python 副作用或性能自动调参。
任务失败沿用依赖执行器行为：停止新提交并排空在途任务，没有外部事务回滚或强制中断。
本次验证的是依赖/时间/数值语义与实际编译能力，没有测得性能加速比。
