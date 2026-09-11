# NGC 核心架构与实现决策

扩展说明：本文保留具体 Pitch 闭环的教学说明。新实现的静态 ECS/World、资产和实验管线见 [World 使用教程](world.md)。

## 1. 范围

Aerodrome 负责飞行器实验，NGC 属于其专业子系统；Probabilistic Aeronautics 的流体和材料课程不依赖这些接口。此原型在独立目录验证架构，后续可迁入原仓库的 `src/aerodrome/`，届时统一 pyproject 和旧版迁移策略。

本次先验证一个小而完整的闭环，不建立通用图形建模平台、动态插件注册中心或巨大的 BaseEnv。教材读者应该能从一个函数看清公式，从一个组合文件看清闭环。

## 2. 目录职责

| 目录 | 负责 | 不应持有 |
|---|---|---|
| core | 信号、时钟、公共数值语义 | 飞机、KF、PID 实例 |
| models | 真实系统、执行器、传感器 | 导航/控制器内部记忆 |
| navigation | 测量校正、输入驱动预测、导航解 | 仿真真值、真实未知参数 |
| guidance | 任务目标到参考指令 | 原始舵机驱动逻辑 |
| control | 参考跟踪、控制器内部状态 | 机体真值 |
| systems | 连接具体组件、定义闭环完整状态 | 文件输出、GUI、MATLAB 会话 |
| runners | 时间循环、批运行 | 特定气动公式与奖励 |
| adapters | 外部系统/后续 Gymnasium 接口 | 原生计算路径中的隐式 I/O |

依赖方向为 core ← 各领域模块 ← systems ← examples/runners 的组合调用。navigation 通过信号使用已知输入，不导入 plant 读取真值。日志作为结果输出，不反过来充当算法输入。

## 3. 数据与算法分离

所有运行中的数值状态使用 NamedTuple，天然作为 JAX PyTree：`PitchState`、`GaussianState`、`NavigationSolution`、`PitchReference`、`PIDState` 等。字符串说明和单位元数据留在文档/宿主配置，不进入每步设备数组。

数值参数使用独立 Parameters，其中 `true_model` 与 `navigation` 的名义离散模型分别设置。修改真实参数不会自动修正估计器。运行步函数签名为：

```python
next_state, record = step(state, goal, parameters)
```

`Schedule` 和 `Blocks` 在构建时固定，由 `make_step` 捕获；数值参数继续作为调用参数，支持后续 vmap 参数扫描。更改组件拓扑/数组形状可能重新编译。

命名信号严格区别：真实 `PitchState`、`PitchMeasurement`、`NavigationSolution`、`PitchGoal`、`PitchReference`、`ElevatorCommand`、`ElevatorPosition`。当前仅有 pitch 测量，KF 估计 pitch 和 pitch rate。

## 4. 接口契约

```text
navigation.correct(prior, measurement, nav_parameters) -> posterior
navigation.predict(posterior, known_input, nav_parameters) -> next_prior
guidance.update(previous_reference, navigation, goal, parameters, dt, tick) -> reference
control.update(controller_state, navigation, reference, parameters, dt) -> (state, command)
models.advance(truth, actuator, physical_parameters, dt) -> next_truth
```

分开 predict 和 correct，让学生看到没有新测量时只是预测，而不是重复校正。当前每个基础步都预测、每个有效传感器样本校正一次。没有实现乱序观测或延迟重放。

控制器状态不能隐藏在对象属性里。PID 的积分记忆显式传递。未来路径规划器、EKF、RNN 或 MPC 需要各自的状态类型；若类型形状改变，则新增明确的 system 组合，不要求套进 `PIDState`。

`Blocks` 支持保持本案例接口的函数替换，例如另一种参考生成函数或同类型控制函数。它不是承诺任意算法可在运行中无缝切换的动态插件系统。

## 5. 一个时间步的严格含义

进入第 k 步时：真实状态与导航 prior 都对应 t_k。

1. 若到传感器时刻，从 t_k 真值产生测量；否则输出 invalid 占位。
2. 导航用当前有效测量校正 prior，产生 t_k posterior/navigation。
3. 若到制导时刻更新参考，否则保持上一参考及其更新时间。
4. 若到控制时刻更新积分状态与指令，否则保持。
5. 理想舵机执行限幅，生成实际输入。
6. 记录 t_k 真值、测量、posterior、参考与本区间输入。
7. 在 [t_k,t_(k+1)) 保持输入，推进真实状态；以已知理想执行量预测导航 prior。
8. 返回 t_(k+1) 状态。

日志有 T 个区间起点，最终状态位于 T*dt，不把终点状态错误地标成上一时刻。首个传感器、制导和控制更新在 t=0；参考发生器首步允许一个制导周期的最大参考增量，日志明确这是离散生成值，不是物理角度跳变。

默认 physics=100Hz、sensor/control=50Hz、guidance=10Hz。这些只是演示值。参数采用整数 tick 周期；未来动态配置需要重新离散化滤波器，而不能只改 dt。

当前舵机是完全已知的无记忆限幅，所以导航预测使用的 actuator 可由 command 确定。新增未知舵机动态时，导航只能使用名义执行器预测或显式编码器测量，不能读取真实执行器内部状态。

## 6. 随机性、精度与输入约束

每次 initialize 接收 root key，传感器按 source ID 与物理 tick 派生 key。改控制更新周期不会改传感器噪声序列；批量环境由调用者给独立 key。初始化的导航先验由用户显式给定，不从隐藏真值生成。

真实系统示例是确定性的；KF 的 Q 表达估计器对未建模角加速度的假设，不是假称系统已注入同等过程噪声。Q 使用连续谱强度离散化，不固定每个 dt 的方差。

入口启用 x64；后续 f32/GPU 精度要独立验证。当前 Schedule 与 KF 离散化检查基础参数，其余 NamedTuple 属于低层数值 API，假定形状正确、数值有限、PID 增益非负、限幅为正、初始协方差正定。公共配置加载器和更完整校验属于下一步，不将此原型描述为已完备的用户输入层。

## 7. MATLAB/FMU 扩展边界

`adapters/external.py` 定义宿主侧协议，现已有 `CoSimulationRunner` 和纯函数分区包装器，见 [World 使用教程](world.md)。真实 MATLAB/FMU 传输适配器尚未实现。

外部组件端口显式声明名称、单位、形状和坐标系。`advance_to` 用零阶保持输入推进到目标通信时刻，并返回真实输出时间；协调器已验证时间、输出、失败状态、重置和资源关闭。超时中断尚未实现。通信步长与内部积分步长分开，外部机体由外部求解器积分。

原生 JAX 路径与外部路径共用信号映射和实验语义，不把 MATLAB Engine 放入 scan。外部 Protocol 不宣称支持 vmap、梯度或 GPU。重置、快照、通信步长、直接馈通能力均由组件显式声明；代数环和回滚未实现。

第一阶段建议先加入 MAT 文件的初值/输入/输出交换与跨实现对照；拿到真实外部模型后再选择 Engine 或 FMU 实现。

## 8. 下一批最合理的工作

正式教学 baseline 已确定为 **F-16 六自由度非线性模型**。当前二状态模型仅用于架构验证，不作为最终飞机模型。

来源推荐已进一步确定为 [ISRL F16-Model-Matlab 的机体/气动实现](f16-selection.md)，发动机与执行器单独选择。可扩展的组合机制见 [模块化管线设计](modular-pipeline.md)；World 装配接口已实现，F16 本身尚未实现。

实施前需选定并记录具体 F-16 模型版本、方程/气动表来源、许可、参数集与适用包线。已有 Aerodrome、NeuralPlane、AeroPlanax 的 F-16 实现可供对照，但不默认等价，也不混合不同版本的气动表与参数。

“完整”在本项目中指清楚定义的工程教学系统：六自由度刚体运动、气动力/力矩、所选模型规定的推进和执行机构动态、大气/重力、配平与验证工况。传感器和 GNC 仍是可替换的外部层。起落架/地面接触、结构弹性、失效模式与高保真流体计算不自动包含在 baseline 承诺中。

课程模型形成同源层次：完整非线性 F-16 → 指定工况配平 → 局部线性化 → 经假设说明与误差验证的纵向/横侧向子模型。线性化模型保留配平状态、配平输入、扰动变量定义、坐标/单位、版本与生成方法，不能将某个工况模型称为全包线近似。

建议新增 `models/f16/`，包含 `parameters`、`aerodynamics`、`propulsion`、`actuators`、`dynamics`；另设 `analysis/trim`、`analysis/linearize`。不将这些内容加入现有 `PitchState`：六自由度模型使用新的命名状态类型与 `systems/f16_loop.py` 组合。

1. 审定 F-16 模型来源、符号、执行器与有效范围，建立模型卡和参考工况。
2. 增加面向用户的配置校验、模型卡和数据记录格式。
3. 引入 LQR/KF 组合与名义/真实参数失配实验。
4. 加入六自由度 `NavigationSolution` 和姿态/航路参考类型，另建六自由度 system。
5. 接入具体 MATLAB 程序，以共同激励和时间轴验证端口。

## 9. 参考

- [JAX 有状态计算](https://docs.jax.dev/en/latest/stateful-computations.html)：显式传递状态。
- [JAX 控制流](https://docs.jax.dev/en/latest/control-flow.html)：固定计算形状、cond 与 scan。
- [JAX 外部回调](https://docs.jax.dev/en/latest/external-callbacks.html)：外部执行与变换限制。

本原型为新写代码，没有复制 NeuralPlane 或 AeroPlanax 的实现。
