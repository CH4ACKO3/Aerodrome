# 本次验证记录

## 2026-10-08：零号工程与五个数学实战工程

完整回归：**172 passed, 36 warnings in 61.69s**。环境仍为 macOS arm64、CPython 3.14.7t、JAX/jaxlib 0.11.1、CPU；告警仍来自既有 python-control 的 NumPy shape 弃用用法。

新增 2 个共享质点模块场景，以及 6 个真实 CLI 端到端场景：三种代码写法等价并与解析离散响应对照，障碍导航、动态跟踪/离散决策、变人数协同、作业调度、效能变化适应。验证训练产物、留出任务、实际状态/动作、事件与导出指标；不按参数拒绝或源码断言新增专项测试。协同与导航的间距指标覆盖步内连续运动，协同另核对成员加入/退出和最终积分区间。

五个项目均实际运行并保存训练数据、模型与全部测试轨迹。导航两法各完成 3/3；协同三法各完成 6/6；故障退化工况末段平均 RMSE 为固定反馈 0.24838 m、在线辨识 0.00048 m、学习估计 0.03788 m。冻结博弈模型的分布变化退化、学习模型在正常工况的额外误差均保留。

详细命令和实现边界见[教学工程目录](teaching-projects.md)。这些是低阶数学模型与小数据方法，不是完整四旋翼/固定翼绕障、覆盖协同、飞行器级故障恢复或强化学习训练结果。零号固定锚点和项目 README 导入已接入教程站构建，项目运行入口当前为 Python CLI。

## 2026-10-08：简单姿态 PD 控制

新增滚转/俯仰 PD、偏航阻尼、空速 P 的六自由度 F-16 示例。实际运行正负初态扰动和固定配平输入对照，控制 50 Hz、物理 100 Hz、20 秒。两条反馈轨迹分别在 8.44 s / 3.44 s 后持续满足姿态、角速度、空速联合误差带；没有高度或航向保持，详细数值见[姿态控制例子](f16-attitude-hold.md)。

新增一个端到端测试，通过真实脚本入口检查闭环恢复、无反馈对照、控制输出和导出文件。完整回归：**164 passed, 36 warnings in 50.29s**，环境同下，告警仍来自已有 python-control 的 NumPy 弃用用法。

## 2026-10-08：六自由度 F-16 与测试整理后的完整回归

macOS arm64，CPython 3.14.7 free-threaded、JAX/jaxlib 0.11.1、CPU、float64，独立 review 虚拟环境 editable 安装，完整 optional dependencies 已安装。`python -m pytest -q`：**163 passed, 36 warnings in 49.53s**。告警仍来自 python-control 的 NumPy shape 弃用用法。

新增覆盖为 4 个完整模块场景与 2 个端到端场景：上游归档运动/六气动系数与惯量耦合、全表 SciPy 插值与纵向退化、六轴配平/风/舵面导数、RK4 收敛，以及真实 CLI 飞行/姿态投影和 World 双轨迹舵面脉冲。未增加断言或参数拒绝分支的专项测试。详细工况、误差标准和来源差异见 [六自由度模型说明](f16-six-dof.md)。

实际 150 m/s、3000 m、4 秒示例完成；配平加速度残差 `1.78e-15`，配平对照高度漂移为零。脉冲实验最大 p/q/r 分别约 20.143/1.383/2.212 deg/s。wheel 和 sdist 构建成功；运行时无需 h5py。未运行新的 MATLAB/Julia 对照或 GPU 测试。

以下保留此前阶段的历史记录与当时的测试数量。

## 2026-09-11 及后续阶段记录

日期：2026-09-11。当前环境：Windows 主机的 Ubuntu WSL2 x86_64、CPython 3.14.7 free-threaded、JAX/jaxlib 0.11.1、CPU、float64。完整依赖快照位于 requirements-tested.txt，安装锁定记录在 uv.lock。

安装：独立虚拟环境中 editable 安装本目录。

测试命令：PowerShell 中 `./scripts/run-wsl.ps1 test`，内部执行 `uv run --locked --extra test python -m pytest -q`。

结果：**172 passed, 36 warnings in 53.25s**。包含 free-threaded 运行时验证、18 项 batch/episode 测试、10 项日志/传感器/性能探针测试、14 项线性模型适配测试、10 项控制工具箱覆盖测试、5 项 F-16 纵向测试、17 项通用刚体测试、10 项公共工具测试、14 项地理/大气测试及 7 项渲染接口测试。36 条告警来自 python-control 在零状态系统上设置数组 shape 的 NumPy 2.5 弃用提示，未屏蔽。旧环境为 CPython 3.12.11 / JAX 0.10.2，已完成向新环境的回归。测试命令现包含 `--extra control`。

`scripts/check_runtime.py` 在启动、导入 NumPy/SciPy/JAXlib/JAX、实际执行 JIT 后均检测到 `sys._is_gil_enabled() == False`，构建标志 `Py_GIL_DISABLED=1`。结果保存在 `artifacts/runtime-314t.json`，未设置强制关闭 GIL 的环境变量。

五个示例已在新环境逐一重跑成功：pitch_tracking、world_pitch、hybrid_propulsion、graph_world、compiled_modules。所有既有数值对照通过，摘要/轨迹已更新；运行未使用 GPU。以下数值与架构验证结论在升级后仍成立。

覆盖 RK4 解析对照与四阶收敛、连续噪声离散化、多速率保持、无新测量不重复校正、隐藏角速度不影响初始导航/控制输入、JIT 与逐步执行一致、批单环境一致、种子与重置、随机源与控制频率解耦、协方差、PID 抗积分饱和以及非法调度参数。

示例命令：`python examples/pitch_tracking.py`

示例结果（1000 步，每步 0.01 s，seed=42）：

| 指标 | 结果 |
|---|---|
| 跟踪 RMSE | 0.0137034083 rad |
| 俯仰角估计 RMSE | 0.0007908224 rad |
| 终点俯仰角 | 0.1007005335 rad |

完整摘要与轨迹在 artifacts/summary.json、artifacts/trajectory.npz。这些数值证明示例可运行，不是控制性能优越性或真实飞机精度的证据。尚未验证 GPU、多平台、MATLAB/FMU 或完整 F-16 六自由度飞机模型；通用固定质量六自由度刚体的验证见下文。

## World 与模块化扩展验证

新增测试验证：

- World 与原闭环在相同初态/随机键下逐步一致；step 等于连续 tick，记录轴与时钟正确。
- 多世界 vmap、实体 ID 随机流隔离和重排稳定性、异构实体状态、共享数值资源。
- 机体参数梯度与中心有限差分对照；联合 PyTree RK4 的四阶收敛。
- 仅替换机体槽位或发动机工厂，其余装配与记录代码可复用。
- Jacobi 分区遍历顺序独立、通信步减半收敛、端口不兼容拒绝、时间错误后禁止继续推进、重置恢复和异常清理。
- 数据哈希错误拒绝、精确版本注册、Pipeline 依赖排序与环拒绝。

`python examples/world_pitch.py`：500 个决策步 × 2 tick = 1000 tick，CPU float64，seed=42。
跟踪 RMSE 为 0.0152295937470 rad，估计 RMSE 为 0.00097497098995 rad。
World 按实体 ID 派生随机流，因此与旧示例相同 seed 并不表示同一噪声序列；回归测试单独对齐了实际随机键。
结果在 `artifacts/world_pitch/`。

`python examples/hybrid_propulsion.py`：100 个通信步，每步 0.02 s。
JAX 发动机与 NumPy 外部协议测试组件在同一机体下运行；两条轨迹的最大绝对差异：位置 1.78e-15 m、速度 5.33e-15 m/s、推力 3.41e-13 N。
结果在 `artifacts/hybrid_propulsion/`。该示例是一维模型，不是 F16，也没有调用 MATLAB。

尚未进行运行性能基准测试。外部接口的超时中断、真实进程通信、快照/回滚和直接馈通代数求解不在本次验证范围。

## 依赖图异步调度扩展

新增 7 项测试覆盖：同 tick 分支并发/汇合、无全局屏障的跨 tick 推进、不同 chunk 的 World 等价性、非零起点时钟、图校验以及失败后的在途任务排空。
并发验证使用 Event 握手，不依赖 sleep 或机器运行速度。

`python examples/graph_world.py` 已运行：两个独立实体，各推进 40 tick，chunk=7，2 个工作线程。
全部轨迹字段与整体 JAX rollout 的最大浮点绝对差异为 0.0。
本次完成记录中 A 的 offset=28 任务早于 B 的 offset=7 任务报告完成，显示两条任务链可以独立推进；完成顺序不作为可重复实验结果。
记录在 `artifacts/graph_world/summary.json`。尚未进行吞吐或加速比基准测试，未验证 GPU 实际内核重叠。

## 模块端口图与跨 tick 编译扩展

新增 11 项测试验证：端口驱动的可执行连线、当前/上一 tick 反馈、宿主每 3 tick 更新时自动分区、宿主路径退出再进入时拒绝跨界融合、独立分区保持、chunk 时间窗口、非零起点与空事件段、融合/不融合对照、输入与时间契约、GraphAssembly 与 World 集成，以及全原生 scan 梯度。

`python examples/compiled_modules.py` 已实际运行，并执行 `native.lower(...).compile()`：

| 验收项目 | 结果 |
|---|---|
| 模型 | 一维速度跟踪、理想观测、比例油门、静态推力映射、Euler 机体 |
| 时间 | 16 tick × 0.02 s |
| 模块 | observation / control / engine / body |
| 原始事件 | 52 次模块更新（engine 每 4 tick 一次） |
| 全 JAX 计划 | 1 个编译区域 |
| 宿主发动机计划 | 9 个区域 |
| 宿主实际调用 tick | 0、4、8、12 |
| 对照 | 全 JAX 区域、宿主分区与完整 native scan 的所有状态/日志通过 rtol=atol=1e-12 对照 |
| 最终速度 | 3.122632671586479 m/s |

完整区域计划保存在 `artifacts/compiled_modules/summary.json`。宿主发动机是进程内 NumPy 测试组件，未调用真实 MATLAB。
任务区域内使用有限图展开；全原生长序列使用 scan。尚未进行编译开销、显存或吞吐基准测试。

## 多 World 与批量 rollout 扩展

新增 18 项测试覆盖逐字段参数轴与共享表、批量/单独执行对照、时间轴与物理子步、紧凑/空记录、稳定 ID 重排和拆分、独立随机重置、无跨 World 梯度项、原生模块图集成、独立终止/截断、末端观测、循环/无状态策略、随机策略跨 chunk 一致性、观测与真值边界、float32 carry 以及非法 ID/mask。

新增示例 `python examples/batch_rollout.py` 已在同一 3.14t CPU 环境运行：64 World × 240 决策步 × 2 tick。不同 World 使用不同的真实阻尼、目标和初始真值，导航器保持标称模型和独立配置的先验。

- 固定输入轨迹投影形状为 `[240, 64]`，最终 tick 为 480。
- 策略观测形状为 `[240, 64, 3]`，只包含导航估计的俯仰角/角速度和任务目标。
- 每 episode 最多 100 步，发生 128 次截断和 0 次任务终止，各 World 最终 episode ID 为 2。
- 平均每步奖励为 -0.002489150036126375；这只是运行记录，不是学习效果指标。示例策略是固定目标的制导基线，没有训练神经网络。

摘要与轨迹保存在 `artifacts/batch_rollout/`。未安装 GPU 插件，本次未验证 GPU 加速、多卡或性能收益。

## 日志、传感器与性能探针扩展

新增 10 项测试验证采样/延迟/保持和 fresh 脉冲、量化/限幅/完全丢包、float32、随机传感器分块与批量等价、参数校验、三轴信号与传感器随机命名空间、延迟队列中丢包后的旧值保持、JIT 投影与 NPZ/JSON 回读、覆盖拒绝与轴校验、线程安全计时和异常记录、融合/非融合计时对照、宿主探针不增加外部调用次数。

`examples/instrumented_rollout.py` 已运行：8 个 World，2 个 chunk，每 chunk 8 个决策步；每个通道数组形状 `[8,8]`。元数据保留 seed、World ID、episode ID、单位和 dtype。读取验证通过，未使用 pickle。

单 World 的 8 tick 逐模块诊断包含 34 次更新：body/control/observation/speed_sensor 各 8 次，engine 2 次。冷/热诊断分别记录，共同生成 74 个时间线事件（另含 batch 编译、预热、两次执行和两次日志写入）。事件与统计均成功导出，未将非融合模块时间声称为融合运行中的独占耗时。

最新本机示例记录：`artifacts/instrumented_rollout/run-cbcf99on/`。CPU 墙钟耗时依赖机器负载，本次不作性能基准或 GPU 结论。

## python-control / 状态空间适配

control 0.10.2 与其依赖 Matplotlib 3.11.1 已在现有 3.14t 环境安装，Slycot 未安装。运行时检查确认导入 control、TF→SS 转换、离散化和阶跃响应后 GIL 仍关闭，报告已更新。

新增 14 项测试覆盖 SISO TF 导入与 forced_response 对照、标签往返、MIMO ZOH/Tustin 和 D 直接馈通、离散模型不重复离散化、编译后禁止调用宿主转换并允许替换矩阵参数、逐 World 梯度独立、模块图/batch/多速率与直接 scan 等价、采样周期与代数环检查、零状态纯增益、float32、时间基准/shape/非有限矩阵/精度/非 proper TF 拒绝、纯数值内核不导入 control/Matplotlib，以及 MIMO TF 的显式边界。

`examples/linear_control.py` 已分别运行 float32 和 float64，对 B=1/64/256、T=512 的完整 World、直接 JAX 与串行 python-control 进行数值对照和计时。基准期间未并发执行本项目测试。float64 最大绝对误差为 3.28e-15；float32 为 1.73e-6，采用显式整轨迹尺度误差预算。

256 World 的 float64 World 热执行中位数约 0.520 ms，直接 JAX 为 0.541 ms，World 冷编译约 57.3 ms。该二维 LTI 测量说明此规模下两条路径处于同一量级，不代表一般系统保证，也没有验证 GPU。完整表格、测量限制及复现方式见[线性模型文档](linear-control.md)，原始记录见 `artifacts/linear_control/benchmark-float32.json` 和 `benchmark-float64.json`。

## Control System Toolbox 覆盖扩展

新增 10 项测试覆盖原生/Matlab 风格分析入口、ZPK/反馈/DC 增益/频率点、LQR Riccati 残差及稳定极点、lsim 返回约定、连续/离散耦合 MIMO TF 与逐通道响应对照、非 proper 通道和状态数上限拒绝、可逆 E 的连续/离散转换、奇异/病态 E 拒绝、零/混合/多步输入延迟、非零历史、batch 与跨 chunk 续算、延迟内存预算。原先 MIMO TF 拒绝测试已改为转换验证。

`examples/control_toolbox.py` 已运行：2 输入/2 输出 TF 转成 3 状态实现，输入时延为 0/3 个模型步，历史缓冲形状 `[3,2]`，64 步输出形状 `[64,2]`。LQR 单状态例题增益为 0.414213562373095，闭环极点为 -1.414213562373095；宿主分析和 JAX 执行后 GIL 为 False。结果在 `artifacts/control_toolbox/summary.json`。

示例性能探针仅演示记录，运行时与回归测试并发，不作为新吞吐基准。没有真实 MATLAB 对照、Slycot 安装或 GPU 验证。[覆盖矩阵与后续验收路线](control-toolbox-coverage.md) 明确保留这些边界。

## F-16 纵向平飞

新增 ISRL/NASA1538 气动表纵向子集、平面非线性动力学、配平/线性化/DLQR 管线及 World Assembly。5 项测试验证查表节点/非节点/域外、数据哈希、体轴投影、配平残差、自动微分/有限差分、Riccati 残差、非线性恢复与 RK4 步长减半。额外核对源 HDF5 的 alpha=0、beta=0、elevator=0 原始节点：Cx=-0.0489、Cz=-0.025、Cm=-0.0598。

3 World × 120 s 非线性仿真已完成：150 m/s、3000 m 配平，alpha=theta 约 3.506°，升降舵 -1.377779°，推力 8401.448 N。正负联合扰动均在 21 s 内进入既定误差带；完整数据和图在 `artifacts/f16_level_flight/`，假设与判据见 [F-16 实验说明](f16-level-flight.md)。这是理想导航与理想执行器的局部纵向基线，无六自由度/MATLAB 对照或 GPU 结论。

## 通用刚体验证

新增 17 项测试全部通过，包括旋转约定的 SciPy 对照、解析平动与外加力矩、完整非对角惯量的无力矩守恒、姿态表示一致性、奇异点行为、子阶段载荷重算、float32/64、World/batch/scan/梯度和 F-16 纵向退化对照。

examples/rigid_body.py 在 CPU/float64 下运行两个 World 各 10 秒，四元数/欧拉角最终旋转矩阵最大差 8.2234e-13；角动量最大绝对误差分别为 2.2579e-14 和 6.5986e-13 kg·m²/s。产物位于 artifacts/rigid_body/。这是数值一致性与基本物理验证，不是完整飞机或 MATLAB 联合仿真验证。


## 公共仿真工具验证

新增 10 项测试通过，覆盖 float32/64 四元数复合与逆、SciPy 旋转向量对照、零旋转 Jacobian、欧拉角运动学导数、刚性变换/力矩/协方差、安装杆臂、ECEF 局部基、风轴与气动角、风速扣除、航迹角、单位和角差。原刚体模型复用公共姿态函数后，原有回归仍通过。结果为 CPU 验证，未测试 GPU 性能。


## 地理位置与大气验证

新增 14 项测试通过：WGS84 坐标、极点/日期变更线、地心距离与高度、局部偏转、1～4 维查表与 SciPy 对照、US1976 分层参考值/反解/连续性、干湿空气、风速、气象时空插值、float32 保持与自动微分。标准大气使用 US1976 气体常数 8314.32/28.9644；未改动旧 F-16 拟合模型。仍为 CPU 验证。


## 渲染数据接口验证

新增 7 项测试通过：JAX 投影/vmap、不可变帧、JSONL 场景和轨迹往返、SLERP 与重采样、在线最新帧覆盖/唤醒/重置/关闭，以及确定性时钟下的 FPS/TPS/RTF。完整回归后新增发布墙钟时间戳和更严格的 tick 单调检查，渲染测试单独复测。

真实 World 示例运行 1000 个物理 tick、dt=0.01，导出含初态 1001 个快照和 601 个 60 FPS 回放帧。故意慢消费时发布 100 帧、覆盖 99 个未读帧，末帧 tick=1000。产物位于 artifacts/rendering_data/20260911T122533747572Z/。没有实际图形呈现，FPS 为 0；示例 TPS 是含宿主传输与文件写入的 CPU 小场景测量，不是渲染性能结论。


## 可替换渲染器后端验证

完整回归 164 项通过；其后补上请求超时逻辑和用例，最终后端专项 10 项通过。覆盖能力协商、生命周期、单请求背压、离线错误、重复回执、呈现计数、线程归属、消息桥和 Three/UE 约定的坐标换基。无头示例完成 61 个离线请求、呈现计数 0；产物见 artifacts/renderer_backend/。未连接真实 Three.js 或 UE。


## 外部设备输入输出验证

完整回归 172 项通过；随后补上鼠标累计边沿及原生图 fresh 去重，最终设备 I/O 专项 8 项通过。测试涵盖输入通道、键鼠事件、模拟摄像头、混合图 source/sink、WorldIO 重放和无效输入门控。实际设备未开启，可选 Pygame/OpenCV 未安装；宿主设备兼容性需另行验证。

## 统一配置验证

### Gymnasium World 适配器复验

接入 Gymnasium 1.3.0 后完整回归 **211 passed，36 warnings，139.31 秒**。新增 9 项适配器测试覆盖官方 check_env、seed/options、观测拷贝、动作拒绝、终止与超时、关闭/失败后的生命周期、直接 World 对照、标准 SyncVectorEnv、Dict 观测/Discrete 动作，以及纯 transition 的 JIT/vmap/梯度。36 条警告仍来自既有 python-control NumPy shape 弃用用法。

刚体速度控制示例在 seed=42 下，196 step / 392 tick（3.92 秒仿真时间）达到任务目标，terminated=True、truncated=False，episode_return=-3.304635464832245。运行环境检查确认 after_gymnasium=False，即 GIL 保持关闭。当前测试设备为 CPU，没有安装训练框架或验证真实图形、AsyncVectorEnv、GPU 训练。

### Hydra 兼容入口复验

接回可选 Hydra 1.3.6 后，完整回归 **202 passed，36 warnings，134.77 秒**。新增 11 项 Hydra 用例：parser 兼容范围和标准库校验保持、应用/框架帮助、补全脚本、配置展开、defaults tree、构建校验、两组 multirun、Pydantic 类型拒绝、自定义配置插值和相对资产路径。multirun 在 `hydra.job.chdir=true` 下验证结果目录、渲染帧数及双向来源记录。

运行环境检查确认 `after_hydra_parser: false`，即 GIL 保持关闭。实际 `aerodrome-hydra scenario=f16 environment=earth runtime.steps=5` 完成 10 tick，结果位于 `artifacts/config_runs/20260911T133056926408Z_2574a987/`。wheel 构建成功并包含两个 CLI 入口和全部配置文件。额外 launcher/sweeper 插件及 GPU 未在本轮验证。

完整回归 **191 passed，36 warnings，61.29 秒**，其中新增配置测试 18 项。警告仍来自 python-control 使用 NumPy 2.5 已弃用的 shape 赋值。验证覆盖严格类型、嵌套覆盖、文件组合、非法字段、资产 hash 与自定义工厂、float32 多实体、CLI 参数扫描、失败状态、渲染采样以及分块运行与 World 直接推进的一致性。

实际 F-16 配置 CLI 完成 5 step / 10 tick，并保存气动数据来源和 hash 清单，产物为 `artifacts/config_runs/20260911T131032504328Z_b9ed0466/`。wheel 构建成功，确认包含全部 7 个 YAML 预设和 console entry point。`artifacts/runtime-314t.json` 确认 Pydantic 2.13.5、PyYAML 6.0.3 配置校验之后 GIL 仍关闭；当前计算设备为 CPU。
