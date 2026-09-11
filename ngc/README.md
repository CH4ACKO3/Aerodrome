# Aerodrome World / NGC 核心架构原型

这是独立的本地设计原型，尚未合并进旧 Aerodrome，也没有推送或发布。

目标是让“导航估计 → 制导参考 → 控制指令”成为可读、可换、可测的代码。基础示例是**假设系数的二状态俯仰模型**；另有基于 ISRL/NASA1538 气动数据的 F-16 纵向子模型实验，尚非完整六自由度飞机。

## 阅读入口

1. [架构与边界](docs/architecture.md)：职责、依赖、时序、扩展方法。
2. [公式到代码](docs/equations.md)：符号、公式、源码和验证对应。
3. [闭环组合](src/aerodrome/systems/pitch_loop.py)：一个基础时间步的全部调用顺序。
4. [可运行实验](examples/pitch_tracking.py)：带噪俯仰角测量、KF、限速参考和 PID。
5. [验证结果](docs/validation.md)：Python 3.14t 下的测试和 CPU 示例结果。
6. [F-16 来源选型](docs/f16-selection.md)：推荐 ISRL/NASA TP-1538 机体与气动来源，已锁定提交和数据 hash。
7. [模块化管线设计](docs/modular-pipeline.md)：资产、组件、组合、场景与训练评测分离；包含 JAX/MATLAB 混合执行边界。
8. **[World 使用教程](docs/world.md)**：已实现的静态 ECS、组件替换、tick/step、批量运行与联合仿真。
9. **[依赖图与异步推进](docs/dataflow.md)**：tick 内显式 DAG、独立实体跨 tick 调度、JAX chunk 与边界。
10. **[模块端口图与跨 tick 编译](docs/module-compiler.md)**：自动连线解析、延迟反馈、宿主边界分区、跨模块/跨 tick JIT 与完整 native scan。
11. **[Python 3.14t 运行环境](docs/runtime.md)**：最新依赖、WSL 入口与实际 GIL 状态检查。
12. **[多 World 与批量 rollout](docs/batch.md)**：共享/逐 World 参数、vmap + scan、独立 episode 重置、策略观测与循环状态。
13. **[日志、传感器与性能探针](docs/telemetry.md)**：命名通道、分块 NPZ、通用采样传感器、融合区域及逐模块诊断计时。
14. **[python-control 与线性模型](docs/linear-control.md)**：TF/SS 适配、宿主离散化、纯 JAX 矩阵执行、批量梯度和性能对照。
15. **[Control System Toolbox 对齐范围](docs/control-toolbox-coverage.md)**：实际覆盖、模型扩展、性能约束与分阶段验收计划。
16. **[F-16 纵向平飞实验](docs/f16-level-flight.md)**：已锁定气动表、非线性纵向模型、配平、DLQR 与批量扰动恢复。
17. **[通用六自由度刚体](docs/rigid-body.md)**：NED/FRD、四元数或 Euler321、合力矩组合、每子阶段载荷与 World 批量积分。
18. **[仿真基础工具](docs/simulation-tools.md)**：坐标/姿态/载荷变换、风轴与气动角、杆臂运动学、单位和角差，兼容 JIT/vmap。
19. **[地理位置与空气物理量](docs/geography-atmosphere.md)**：经纬高、地心距离、高度基准与查表、相对方向、标准大气、温压湿度与气象网格。
20. **[离线/在线渲染数据接口](docs/rendering.md)**：Scene/Frame、JAX 投影、JSONL、SLERP 回放、最新帧缓冲与独立 FPS/TPS/RTF。
21. **[可替换渲染器后端](docs/renderer-backends.md)**：生命周期、能力协商、消息桥、相机/部件动画、回执及 Three/UE 坐标边界。
22. **[外部设备输入输出](docs/external-device-io.md)**：键鼠、摄像头、WorldIO、图内 host/JAX 输入模块、新鲜度与输出通道。
23. **[统一配置与实验运行](docs/configuration.md)**：Pydantic + YAML、可选 Hydra 3.14t 兼容入口、配置组、参数扫描和实验快照。
24. **[Gymnasium World 适配器](docs/gymnasium.md)**：独立环境接口、共享 Task、种子与 reset、动作映射、终止/超时及原生 JAX transition。
25. **[三平台构建检查](docs/platform-builds.md)**：Windows/Linux/macOS 实机验证、独立 wheel 检查及 Windows 3.14t 依赖限制。

## 安装与执行

使用 **CPython 3.14.7t（free-threaded）**，版本由 `.python-version` 固定，依赖由 `uv.lock` 锁定。当前 Windows 主机通过 Ubuntu WSL 运行：

```powershell
./scripts/run-wsl.ps1 check
./scripts/run-wsl.ps1 test
./scripts/run-wsl.ps1 examples
```

独立 Linux checkout 使用最新 uv，在本目录执行：

```shell
uv sync --locked --extra test --extra control
uv run --locked --extra control python scripts/check_runtime.py
uv run --locked --extra test --extra control python -m pytest -q
uv run --locked python examples/compiled_modules.py
```

基础依赖安装 CPU JAX，示例不要求 GPU。JAXlib 0.11.1 没有 Windows 原生 cp314t wheel，因此本机使用 Linux WSL，不能用普通 Windows Python 3.14 替代无 GIL 构建。应用入口选择 x64，库导入不会改变设备、全局精度或 GIL 状态。依赖快照见 `requirements-tested.txt`。

示例在当前目录 `artifacts/` 写入摘要与轨迹。无测量的时间点用 `measurement_valid=False` 表示，绘图时须使用这个 mask，不能把占位的零当作真实测量。

## 已实现

- 命名信号、显式算法状态、静态多速率时钟。
- 连续二状态模型/RK4、理想限幅舵机、独立随机源的俯仰角传感器。
- 线性 KF（Joseph 协方差更新）、限速参考发生器、带条件积分的 PID。
- 同一闭环的 Python 循环、JAX scan、JIT 和 vmap 验证。
- 外部求解器的宿主侧端口与组件 Protocol。
- 静态 Entity/Assembly/WorldSpec 与显式 WorldState；共享数值资源、tick/step/rollout、独立随机流。
- Assembly 执行计划校验、可替换传感器/控制器/机体等槽位、通用 PyTree RK4。
- Registry 精确版本工厂、Asset 哈希校验、独立 Scenario/Evaluation/Pipeline。
- Jacobi/ZOH 联合仿真协调器、原生函数分区适配器、端口/时间/失败恢复检查。
- 两个新增可运行示例：World 俯仰实验；JAX 发动机与外部协议测试组件替换。
- DependencyExecutor 通用 DAG，以及 GraphWorldRunner 独立实体跨 tick 的并发执行和轨迹对齐。
- 可执行 ModuleGraph、端口/时间边校验、自动 JAX 区域融合、GraphAssembly 接入 World，以及全原生图的 scan/梯度入口。
- BatchedWorld：逐字段参数轴、共享资源、批量 tick/step/rollout、紧凑记录、稳定 ID 随机流与独立重置。
- EpisodeRunner：任务观测边界、终止/截断、自动重置、无状态/循环策略和跨 chunk 续算。
- Recorder/ChunkWriter、SampledSensor 与线程安全 PerformanceProbe；编译区域计时、关闭融合后的逐模块计时和 trace 导出。
- 可选 python-control 原生/Matlab 风格分析入口、SISO/MIMO TF 与 SS 导入、ZOH/Tustin 离散化、纯 JAX LTI Module。
- 有条件的描述系统 E 消元、精确整数步输入时延环形缓冲、MIMO 状态数与时延存储预算检查。
- 固定质量六自由度 Newton–Euler 刚体、可选姿态表示、完整惯量张量、JAX 载荷回调与 RigidBodyAssembly。

## 后续扩展

正式教学 baseline 采用 F-16 六自由度非线性模型。课程中的局部线性和纵向/横侧向模型将从选定版本的 baseline 配平与线性化得到，并注明简化假设和有效范围；当前二状态示例仍仅用于架构验证。

完整飞机模型、真正的航路制导、INS/GNSS/EKF、丢包/延迟、LQR/MPC/RL、配置文件加载、完整实验产物管理以及真实 MATLAB/FMU 传输适配器都尚未实现。动态发动机只有一维协议演示，不是 F16 发动机。

World 的 Assembly 接口接受不同状态 PyTree；Pitch 的专业信号仍只适用于二状态例子。不要让所有估计器或不同维度模型继承这个二状态数据结构。材料/流体课程使用独立领域接口，最多共享实验记录与统计工具。
