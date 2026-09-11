# Control System Toolbox 对齐范围与实施路线

目标是尽量缩小本科/研究生控制教学的工作流差距，并逐步覆盖工程需求。不是宣布当前已经替代 MATLAB Control System Toolbox，也不是要求所有分析函数都在 GPU 上运行。

对照日期：2026-09-11。基准是 MathWorks 的 [Control System Toolbox 功能目录](https://www.mathworks.com/help/control/index.html)；上游复用 [python-control 功能目录](https://python-control.readthedocs.io/en/latest/functions.html)和 [MATLAB Compatibility Module](https://python-control.readthedocs.io/en/latest/matlab.html)。当前未连接真实 MATLAB，因此数值对照主要来自解析解、SciPy 和 python-control，不能把这些测试称为 MATLAB 等价性认证。

## 三种支持程度

1. **宿主分析可用**：可通过上游 API 建模、响应分析、设计、画图。沿用其参数、返回轴和限制，不把函数同名当作语义等价。
2. **JAX 执行已适配**：能显式转换为稳定 shape/dtype 的运行模型，进入 ModuleGraph、batch、rollout 和梯度路径。
3. **待开发或未验收**：需要新求解器、工具链或专门数值验证；不会静默降级、增加延迟、取消极零点或拟合替代。

## 当前覆盖矩阵

| 功能 | 当前路径与限制 | 下一步 |
|---|---|---|
| 连续/离散 SS，SISO TF | 宿主建模、转换与 JAX 执行已有测试 | 更多病态/高阶测试 |
| proper MIMO TF | 新增逐通道精确实现，不依赖 Slycot；一般非最小 | 显式最小实现/降阶与误差报告 |
| ZPK | 上游 `zpk` 生成 TF，再走转换 | 标签与尺度回归 |
| FRD 频响数据模型 | 上游可分析；不直接进入时域仿真 | 显式拟合接口及误差带，不能自动推断实现 |
| 描述系统 E dx/dt=Ax+Bu | 新增可逆、条件数受限 E 的求解消元；保留 x 坐标 | 奇异 E 的正则性、DAE 指标与一致初值求解 |
| 非 proper TF | 标准执行接口仍拒绝 | 描述系统/输入导数路径，不偷偷加滤波 |
| 离散整数步输入时延 | 新增逐输入精确环形缓冲，含初始历史、batch 和分块续算 | 输出/内部延迟及独立组件接口 |
| 连续/分数步时延 | 上游 Padé 可显式近似；不是原生精确时延模型 | 时延系统表达、历史插值与误差控制 |
| 串联、并联、反馈、命名互连 | 复用上游组合；组合成 SS 后可编译 | 更多直接馈通/奇异闭环测试 |
| step/impulse/initial/lsim | 上游分析；JAX 路径已有固定采样输入 rollout | 统一时间/输出轴转换与指标报告 |
| Bode/Nyquist/根轨迹/稳定裕度 | 上游接口开放；没有逐项完成本项目验收 | 标准教学例题、图与数字回归 |
| 极点、零点、DC 增益、可控/可观 | 复用上游；部分高级 MIMO 数值算法依赖 Slycot | 算法/依赖能力探测与病态矩阵测试 |
| place、LQR/DLQR、LQE/DLQE | 上游可用的设计路径，部分可指定 SciPy；本轮验收一个 LQR 例题 | 离散估计器接入与完整 LQG 教学闭环 |
| Lyapunov/Riccati | 复用 SciPy/上游 | 残差、对称性、稳定性和条件数的统一报告 |
| PID、PID 自动整定 | 当前有专用俯仰 PID；不是通用 `pid`/`pidtune` 兼容实现 | 通用 PID/PIDF、显式整定目标、失败诊断 |
| minreal、平衡截断、降阶 | 上游存在部分功能，部分需要 Slycot；尚未验收 | 独立验证依赖与误差界，禁止隐式降阶 |
| 模型数组、参数扫描 | 同结构 JAX batch 已有；异构阶数不能直接混批 | 分组执行、标称模型与参数范围元数据 |
| 多目标自动整定、增益调度、可调块 | 尚无完整 `systune` 类实现 | 单独优化模块与约束/收敛报告 |
| 交互式设计 App | 没有 MATLAB App 等价物 | 教程站与可导出交互设计实验 |

MATLAB 的描述系统范围包含更一般的 E；可逆消元不能覆盖全部 `dss`，见 [官方 dss 文档](https://www.mathworks.com/help/control/ref/dss.html)。其延迟模型也涵盖输入、输出和内部时延，见[官方延迟说明](https://www.mathworks.com/help/control/ug/time-delays-in-linear-systems.html)。本项目没有把这两类广泛能力等同于本轮新增子集。

Robust Control、System Identification、Model Predictive Control、Simulink/Simulink Control Design 属于应另外确认的产品/范围；不混入本表计算覆盖率。以后有需要可以接入，但不能仅凭 python-control 中有某些相关函数就声称完整替代。

## 本轮使用方式

可运行例题：`uv run --locked --extra control python examples/control_toolbox.py`。完整测试已通过 102 项，具体范围见[验证记录](validation.md)。

```python
from aerodrome.analysis import python_control, matlab_compat, backend_status
from aerodrome.adapters.control import from_control, from_descriptor
from aerodrome.adapters.linear_delay import with_input_delay

ct = python_control()       # 懒加载原生 Python 接口
ml = matlab_compat()        # 懒加载上游 MATLAB 风格接口
G = ct.zpk([], [-1], 2, dt=0)
closed = ct.feedback(G, 1)
model = from_control(closed, sample_time=0.01)
delayed = with_input_delay(model, steps=3)
# delayed.parameters / delayed.initialize / delayed.module 接入方式与基本 LTI 相同
```

`ml.lsim` 返回次序和时间轴遵循 python-control 的 matlab 模块，不能和原生 `forced_response` 返回对象混用。`backend_status()` 只说明安装版本和边界，不表示每个函数在本环境都已验收。缺少 Slycot 的函数仍会按上游规则失败，当前没有自动安装/替换算法。

## 保持性能的约束

- **转换只做一次**：MIMO TF、描述系统消元和离散化放在宿主准备阶段，热路径仍为原有 JAX 矩阵运算。
- **显式限制阶数**：MIMO 逐通道实现的阶数可能达到各通道阶数之和，即使共享极点也不自动合并。默认 `max_states=256`，超限要求紧凑 SS 或用户明确提高预算；`conversion_notes` 记录非最小实现。高阶系统优先提供物理 SS。
- **时延不扩成稠密大矩阵**：保存 `[max_delay,nu]` 环形历史，状态存储量线性增长，矩阵 A/B/C/D 不变；默认 `max_buffer_values=1_000_000` 限制每 World 历史标量数。批量内存还需乘 B。时延缓冲长度会影响编译 shape。
- **按模型采样计时延**：`steps=3` 表示三个模型更新周期。若 Module.every=4，即 12 个 physics tick。正延迟默认零历史；可传 `history`，行按最旧到最新排列。零延迟通道保留 D 的当前输入馈通。
- **不强迫分析函数进入 JAX**：设计、可视化和病态矩阵诊断允许用宿主算法；可微/批量需求有明确收益时再做 JAX 专用实现。
- **不隐式近似**：奇异/病态 E 明确拒绝，不使用伪逆；时延不取整；非 proper 系统不加滤波；FRD 不自动拟合；降阶和 Padé 必须由用户显式选择。

`InputDelayedModel` 含外部于 A/B/C/D 的显式历史状态，不能直接交给 `to_control` 并忽略缓冲；如需导出完整带时延模型，应另做有预算的增广状态转换。

## 后续实施顺序与验收标准

**下一阶段优先完成常用教学闭环**：通用 PID/PIDF、place/LQR/LQE 的运行适配、时域指标、频域响应/裕度报告，以及闭环直接馈通组合测试。用可控性、稳定性、Riccati 残差和解析例题验收，不只检查函数是否能调用。

**随后补模型表达与数值难点**：输出/内部延迟、连续时延、描述系统、最小实现和降阶。每种近似必须保存方法、阶数/容差、有效频段和对照误差。

**最后推进自动整定与教学交互**：明确跟踪、扰动抑制、裕度等目标；报告失败/局部最优；建立可复现实验，而不是把任意优化器包装成 MATLAB 自动整定的等价替代。

建立逐功能验收清单；只有满足语义、数值、单位/时间轴、批量/梯度（适用时）、性能与失败模式测试的条目才提升为“本项目已验证”。真实 MATLAB 对照属于后续独立步骤；未获得其运行结果前不宣称完全兼容，也不随意给出覆盖百分比。
