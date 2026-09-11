# python-control 与高性能线性模型

使用 python-control 定义/分析系统，转换后只让 JAX 数值矩阵进入 World。没有在每个 tick 调用 python-control、SciPy、矩阵指数或 TF 转换，也没有在设备内执行 Python 回调。

## 安装与入口

python-control 是可选依赖，锁定版本为 0.10.2：

```sh
uv sync --locked --extra test --extra control
uv run --locked --extra control python examples/linear_control.py
uv run --locked --extra control python examples/linear_control.py --dtype float32
```

Windows 的 WSL launcher 会安装和使用 control extra。基础 JAX 数值内核不导入 control 或 Matplotlib；未安装 extra 时仍可通过已有 SciPy 的 `from_matrices` 构建模型。Slycot 不作为依赖。

## 从传递函数到可执行矩阵

```python
import control as ct
import jax
from aerodrome.adapters.control import from_control, from_matrices, to_control

jax.config.update("jax_enable_x64", True)  # 或显式请求 float32
plant = ct.tf([3.0], [1.0, 0.8, 0.4], dt=0)
model = from_control(plant, sample_time=0.01, method="zoh")
parameters = model.parameters  # LinearParameters(A, B, C, D)，JAX 数组
discrete_control_model = to_control(model)  # 返回宿主侧供 Bode/LQR 等分析
```

支持 proper SISO/MIMO 传递函数与 MIMO 状态空间。SISO TF→SS 明确选择 SciPy 方法；MIMO TF 使用逐通道实现，不依赖 Slycot，一般非最小，默认状态数上限为 256。`conversion_notes` 标明实现边界；高阶系统建议提供紧凑 SS。纯增益可以用零状态维的状态空间表示。当前数值执行只接受有限实矩阵和 float32/float64。描述系统、时延和分析入口的扩展见[工具箱覆盖表](control-toolbox-coverage.md)。

`python-control` 的连续模型必须明确 `dt=0`，离散模型必须给出正采样时间；`dt=None` 或 `dt=True` 的未确定时间基准会被拒绝。相比之下，本项目 `from_matrices(..., dt=None, sample_time=...)` 明确表示输入连续矩阵；`dt>0` 表示已经离散化，不能再用不同的 sample_time 隐式重采样。

连续系统支持 ZOH 和 bilinear/Tustin 离散化，均在宿主设置阶段执行一次。ZOH 对分段常值输入给出精确离散 LTI 模型；Tustin 使用双线性变换，两者的输入/状态含义与近似性质不同，不应混作同一种积分方法。该转换来自 [SciPy cont2discrete](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html)。TF 转换边界见 [python-control tf2ss](https://python-control.readthedocs.io/en/latest/generated/control.tf2ss.html)。

传递函数转换得到的状态通常是实现坐标，不等于真实俯仰角/角速度。若教学需要物理状态解释，应直接提供物理意义明确的 A/B/C/D；F16 配平点附近的线性模型还应单独保存状态/输入偏移、坐标与单位，当前适配器不会推断它们。

## 输出时序与模块接入

运行方程为：

\[
y_k=Cx_k+Du_k,\qquad x_{k+1}=Ax_k+Bu_k.
\]

`step` 返回 `(x_next, y_current)`，不会用更新后的 x 计算 y。输出记录属于当前采样时刻，最终状态则属于下一采样时刻。

```python
from aerodrome.adapters.external import Port
from aerodrome.composition.module_graph import ModuleValue

module = model.module(
    "plant",
    Port("u", "rad", (1,), "body"),
    Port("y", "rad", (1,), "body"),
    every=1,
)
x0, y0 = model.initialize()  # 默认 x0=0, u0=0
initial_module = ModuleValue(x0, {"y": y0})
# 把 module 放入 ModuleGraph，initial_module 放入对应模块初态，
# model.parameters 放入 graph/World 的显式参数树。
```

端口一律使用 `[nu]` 和 `[ny]` 向量，包括 SISO。多种物理量组成的 MIMO 向量需要明确逐通道单位；不要给混合单位向量随意标一个统一物理单位，可以在外围添加拆分/归一化模块。模型保存 python-control 的输入、输出和状态标签，端口单位与坐标系仍由用户声明。

图编译时检查 `every * physics_dt_s == model.dt`。每个模块只在自身计划采样点推进；其余 tick 保持上次输出。初始化的 y0 供首次更新前/延迟连线使用；如逐 World 的 C/D 或 x0/u0 不同，应把对应参数传给 `initialize(..., parameters=...)`，构造匹配的初态。

直接馈通 D 会保留。现有图仍将模块视为原子更新，保守拒绝同 tick 的环，即使某个状态空间块 D=0，也不会自动拆分“输出”和“状态更新”来消环。闭环可以在宿主 python-control 中先组合为一个完整状态空间块再导入；只有确有采样延迟的连接才能显式加 delay=1，适配器不会偷偷增加延迟。

## batch、梯度和缓存

模块执行函数只捕获 shape、dtype、端口名和采样周期；A/B/C/D 始终是动态参数。在相同 shape/dtype 下，替换矩阵数值可以复用已编译 executable，无需重建模块。拓扑、状态阶数、batch 大小、采样调度或 rollout 长度改变则可能需要重新编译。

使用 `BatchedWorld.parameter_axes` 选择共享或逐 World 矩阵。例如 `LinearParameters(None, 0, None, None)` 表示只让 B 具有 World 维。不要为共享矩阵显式复制 B 份；矩阵转换也不要放入 per-World/per-tick Python 循环。

数值内核单独位于 `models.linear`：

```python
from aerodrome.models.linear import rollout
run = jax.jit(jax.vmap(rollout, in_axes=(0, 1, None), out_axes=(0, 1)))
# x0: [B,nx]，inputs: [T,B,nu]，parameters: 共享矩阵
final, outputs = run(x0, inputs, parameters)
```

JAX 可以对离散矩阵、输入和初态求导，支持 batch/scan 的反向传播。宿主 TF→SS 和离散化过程不是 JAX 图的一部分，因此不能直接通过它们对原始连续模型系数求导。若需要这种训练方式，应另加 JAX 可微离散化/连续积分路径；当前不声称已经支持。

## 性能验证方法

`examples/linear_control.py` 比较三种路径：完整 BatchedWorld+GraphAssembly、直接 JAX 矩阵 scan、逐系统 python-control `forced_response`。全部使用相同输入、初态及输出采样时刻，先核对轨迹，再分别报告：

- 宿主模型准备、输入设备放置、编译和就绪后输出读取耗时。
- 预热后 5 次 JAX 同步执行的原始值与中位数。
- 预热后 3 次串行 python-control 参考运行的中位数。

默认 B=1/64/256、T=512，支持 float32/float64。输出仅保留 y，不保留全状态物理日志；初始化和文件写入不计入热执行。计时前执行依赖已准备好，每次计时都等待返回结果完成。运行期间不要同时执行测试或其他基准。

串行 python-control 是数值参考，不是优化后的批量基线；小二维 LTI 的吞吐不能外推到 F16、复杂传感器或 MATLAB 联合仿真。判断本项目封装开销主要应看“完整 World”与“直接 JAX”的对照，并保留计时波动。本次只在 CPU 上验证，GPU 和多卡尚未实测。

2026-09-11 本机 CPU / Python 3.14t 实测，512 步，热执行中位数：

| 精度 | World 数 | 完整 World | 直接 JAX 内核 | World 编译 |
|---|---:|---:|---:|---:|
| float64 | 1 | 0.126 ms | 0.103 ms | 61.9 ms |
| float64 | 64 | 0.323 ms | 0.342 ms | 60.4 ms |
| float64 | 256 | 0.520 ms | 0.541 ms | 57.3 ms |
| float32 | 1 | 0.098 ms | 0.076 ms | 70.9 ms |
| float32 | 64 | 0.255 ms | 0.331 ms | 59.9 ms |
| float32 | 256 | 0.452 ms | 0.500 ms | 55.6 ms |

单系统封装/调用开销较明显；批量情况下两条 JAX 路径处于同一量级，微小差异包含测量波动与编译布局差异，不能据此断言 World 必然更快。冷编译远贵于一次热执行，所以应复用同一 executable、多步一起运行。

相对 python-control 参考的最大绝对差异：float64 为 `3.28e-15`，float32 为 `1.73e-6`。python-control 参考计算提升到 float64；float32 在过零附近不能仅用逐点相对误差判定，因此这个示例使用 `64 * eps(float32) * max(1, max(abs(reference)))` 的整轨迹尺度绝对限值，本次约 `8.61e-6`。这只是本例的验证预算，不是任意系统阶数、条件数和时长的误差保证。

原始计时、误差限与环境信息保存在 `artifacts/linear_control/benchmark-float32.json` 和 `benchmark-float64.json`。
