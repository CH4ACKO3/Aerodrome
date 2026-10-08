# 静态 ECS 与 World 使用教程

状态：已实现的本地原型。入口为 `examples/world_pitch.py` 和 `examples/hybrid_propulsion.py`。
原有两个示例分别使用二状态俯仰模型和一维推进模型。后续已加入 [F16 六自由度机体示例](f16-six-dof.md)；真实 MATLAB Engine/FMU 适配器仍未实现。

新增 [依赖图执行器](dataflow.md)：保留本页纯函数 World 接口，并允许独立实体在宿主任务图上跨 tick 推进，最终返回同一格式的对齐快照。

## 1. 对象与职责

| 对象 | 实现 | 职责 |
|---|---|---|
| Entity | `composition.world.EntitySpec` | 稳定 ID 与 Assembly |
| Component state | 任意固定结构的数值 PyTree | 真实状态、滤波器、控制器记忆等 |
| System | 领域函数 + `SystemSpec` | 可查阅的计算、输入输出、状态所有权与方程标识 |
| Assembly | `Assembly` Protocol / `PitchAssembly` | 初始化与显式闭环组合 |
| WorldSpec | 静态实体、Schedule、ticks_per_step | 构建配置，不持有运行状态 |
| WorldState | tick、实体状态 tuple | 纯函数推进的唯一动态状态 |
| WorldParameters | 实体参数 tuple、共享数值 resources | 动态函数输入，允许参数扫描和微分 |
| Registry / Asset | `catalog` | 精确版本工厂、来源/许可/文件哈希 |
| Scenario / Pipeline | `pipelines` | 场景、实验依赖、评测与产物生成 |

ECS 在这里提供组织方式。GPU 性能来自可编译的数组计算和批处理，需要针对实际模型测量。
没有运行期实体查询或动态组件增删。显式端口图的自动编译由新增 [ModuleGraph 编译器](module-compiler.md) 提供，可通过 GraphAssembly 接入 World。

## 2. 运行第一个 World

在项目目录执行：

```shell
python examples/world_pitch.py
python examples/hybrid_propulsion.py
python -m pytest -q
```

已有 `world_pitch.build_experiment()` 返回 World、Scenario 和指标函数，可在 notebook 里复用：

```python
from world_pitch import build_experiment  # 将 examples/ 加入 Python 模块路径
import jax

world, scenario, metrics = build_experiment()
state = world.reset(scenario.seed, scenario.initial_conditions)
inputs = world.pack(scenario.inputs)
parameters = scenario.parameters

world.validate(state, inputs, parameters)
next_state, tick_record = world.tick(state, inputs, parameters)
next_state, step_trace = world.step(state, inputs, parameters)
compiled_step = jax.jit(world.step)
next_state, step_trace = compiled_step(state, inputs, parameters)
```

最后三个调用从同一 `state` 出发；它们不会修改它。参数必须显式传入，避免把全部数值配置捕获成编译常量。
`World.pack` 把按实体名称填写的字典转换成固定顺序 tuple；缺失或多余 ID 会报错。

交互式可用 `SimulationSession(world, state, parameters).step(inputs)`，由宿主包装器保存最新状态。
Session 不参与 JAX 变换，也不负责隐式录下无限长历史。

## 3. 时间与记录

`tick` 推进一个 `physics_dt_s`；`step` 推进 `ticks_per_step` 个 tick，并在整个决策区间保持外部输入。
`step_dt_s = physics_dt_s * ticks_per_step`。
它与控制更新周期是两个独立配置；示例将两者对齐。多速率触发始终依据全局 tick，跨 step 不会重新计时。

PitchAssembly 接受的外部输入是 **PitchGoal**，不是裸舵偏。
将 RL 放在制导或控制层需要定义对应输入类型与新 Assembly，不能仅把一个数组改名为 action。

记录定义：

| 调用 | 日志时间轴 | 最终状态 |
|---|---|---|
| tick | 单个区间起点 t_k | t_(k+1) |
| step | `[physics_tick, ...]` | 决策区间终点 |
| rollout | `[decision_step, physics_tick, ...]` | 全部区间结束 |
| vmap(rollout) | `[world_batch, decision_step, physics_tick, ...]` | 每个独立 World 的终点 |

上表最后一行是直接对旧 `World.rollout` 做 vmap 的轴顺序。新的 `BatchedWorld.rollout` 使用时间优先的 `[T, B, S, ...]`，提供参数共享、独立重置和策略执行入口，见[批量教程](batch.md)。

`WorldRecord.entities` 是按 ID 排列的 tuple，不是实体数组轴；不同实体可以有不同状态结构。
每个实体内部的积分、传感器、导航、制导与控制顺序由 Assembly 明确编写。
`SystemSpec` 校验显式计划的依赖、缺失信号、多重写入和状态所有权，但不会从元数据生成算法，不能证明任意用户函数没有隐藏副作用。

连续强耦合状态放入一个 PyTree，由 `core.integrators.rk4` 在每个中间阶段重新求联合 RHS。
外部会话不能作为该 RHS 的有状态子调用。

## 4. 自定义组件与 Assembly

同类型替换可以复用现有状态和连线：

```python
from dataclasses import replace
from aerodrome.systems.pitch_loop import Blocks
from aerodrome.composition import PitchAssembly

assembly = PitchAssembly(blocks=replace(Blocks(), control_update=my_controller_update))
```

可替换槽位包括导航预测/校正、制导、控制、传感器、无记忆舵机和机体推进。
每个函数必须遵守原槽位签名和信号语义。例如换成未知动态舵机后，原先的已知理想执行量预测假设不再成立，需要新的闭环组合。

不同维度的 F16、EKF、发动机或有记忆执行器，应建立新的状态类型和 Assembly：

```python
class MyAssembly:
    backend = "jax"
    systems = (...)           # SystemSpec 的显式执行顺序
    initial_signals = (...)

    def initialize(self, initial_conditions, key):
        return my_initial_state(initial_conditions, key)

    def make_tick(self, schedule):
        def tick(state, inputs, parameters, context):
            # context: 世界 tick、物理 dt、共享数值资源
            return my_next_state, my_record
        return tick
```

World 不读取实体真值来初始化导航。PitchInitial 要求分别给出 truth 和 navigation_prior。
随机键由 seed 和实体 ID 的稳定 SHA-256 派生，增添/调整其他实体顺序不会改变已有实体随机流；32 位流 ID 冲突在构建时报错。
改实体 ID 会改变随机流。与原始 pitch 示例相比，World 增加实体命名空间，因此即使 seed 同为 42，噪声轨迹也不同。

同一 World 的实体目前独立推进，共享 resources 是只读数值输入。机间感知、碰撞或交互尚未实现。
以后应显式增加从同一时刻世界快照读取的交互阶段，避免实体遍历顺序引入偏差。
运行中动态增删、active 掩码槽位管理、自动按 archetype 分组属于后续扩展。

## 5. 外部求解器与混合执行

`CoSimulationRunner` 管理宿主会话；`build_world` 当前只构建原生 JAX World，遇到外部后端会明确报错。
两者具有不同的状态所有权，没有伪装成可纯函数复制的统一状态。

```python
with CoSimulationRunner(
    components={"engine": engine, "body": body},
    connections=(Connection(("engine", "thrust"), ("body", "thrust")),),
    communication_dt_s=0.02,
    external_inputs=(("engine", "throttle"),),
) as runner:
    runner.initialize({"engine": engine_parameters, "body": body_parameters})
    samples = runner.step({("engine", "throttle"): throttle_value})
```

组件使用 `adapters.external.ExternalComponent` 契约。`FunctionalComponent` 能将纯 JAX 子系统包装成宿主分区；进程句柄仍只属于外部适配器。
`hybrid_propulsion.py` 只替换 engine 工厂，对比 JAX 发动机与 NumPy 外部协议测试组件。它不调用 MATLAB。

当前实现：

- 构建时检查端口是否存在、必需输入唯一来源、shape/dtype/单位/坐标系/物理量/参考点匹配。
- 每个通信区间开始，先复制所有旧边界输出，再零阶保持输入独立推进各分区。
- 初始化输出必须由初始状态/配置确定。直接馈通组件全部拒绝；需显式锁存或重新划分子系统。
- 检查输出字段、有限数值、dtype/shape 和实际停止时间（绝对容差 1e-10 s）。
- 中途失败可能已推进部分组件，因此禁止继续 step；只有全部重置成功才允许恢复。
- context manager 退出时关闭全部组件；无快照/回滚、超时中断或迭代耦合。

原生 World 记录区间起点；外部 sample 是通信终点。比较轨迹时必须明确对齐时间戳。
共享初始化时间由协调器提供，reset 必须回到同一起点。
内部子系统即使使用高阶或精确积分，整体 ZOH 分区耦合仍需做通信步长收敛验证。
整个宿主管线不能 JIT/vmap/grad；内部 FunctionalComponent 的 transition 可以单独 JIT。

## 6. 数据与实验管线

Registry 按 `(model_id, version)` 精确查找工厂，重名拒绝，不执行配置字符串。
Asset 记录身份、来源、许可、路径、哈希和元数据；`read_verified()` 返回通过哈希检查的原始字节。
资产解析、单位转换和放入设备资源的操作仍由模型加载器显式负责。尚未实现通用 HDF5/F16 加载器和配置文件解析器。

Scenario 包含 seed、初始条件、外部输入、数值参数和决策步数；不持有求解器会话。
`evaluate` 执行原生 World 并调用独立指标函数。`Pipeline` 在运行前检查阶段依赖和环，只传递声明的依赖产物。
示例执行 `scenario → evaluate → report`。训练阶段可以在自己的函数中执行 rollout/update 循环。

当前 Pipeline 是顺序、内存中的实验 DAG，没有缓存、分布式执行、自动策略训练或完整 provenance 数据库。
示例 JSON 记录时钟、种子、模型声明、平台和指标；完整源代码/资产/展开参数锁定记录仍需后续补齐。

## 7. 本次验收

测试覆盖原闭环等价、tick/step 时间轴、多速率保持、独立世界 vmap、实体随机流稳定性、异构状态与共享资源、参数梯度对照、联合 RK4 收敛、外部分区遍历顺序独立、通信步收敛、端口错误、失败恢复与关闭，以及资产校验和 Pipeline 依赖校验。

`models/f16.py` 和 `composition/f16.py` 的 `F16Assembly` 已按此协议接入，复用独立的 `RigidBodyState`。`examples/f16_six_dof.py` 展示配平与三舵面脉冲的 World 批量运行。
