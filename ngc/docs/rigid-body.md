# 通用六自由度刚体动力学

`models/rigid_body.py` 实现固定质量、固定惯量刚体；`composition/rigid_body.py` 将其接入现有 World、BatchedWorld、scan rollout。它不依赖某一架飞机的气动表。当前 F-16 平飞示例仍使用纵向简化模型；测试验证其方程是通用模型在对称运动假设下的退化形式，尚未补齐 F-16 横侧向气动与发动机。

## 坐标与状态

地面采用局部惯性 NED（北、东、下），机体采用 FRD（前、右、下），均为右手系；位置在质心。内部统一 SI 和弧度，输出可用 `jnp.rad2deg` 转成度。

| 字段 | 含义 |
|---|---|
| `position_ned_m` | NED 位置，形状 `(3,)`；高度为负的 down 分量 |
| `velocity_body_m_s` | 对地速度在机体系中的分量，不自动等于相对空气速度 |
| `attitude` | 四元数 `(4,)` 或欧拉角 `(3,)` |
| `omega_body_rad_s` | 机体系角速度 `[p,q,r]`，不等于一般情况下的欧拉角导数 |

`RigidBody6DoF(attitude="quaternion")` 是默认配置。四元数使用 Hamilton 乘法、标量在前 `[w,x,y,z]`，其旋转矩阵 `R_nb` 将机体系向量变换到 NED。`attitude="euler321"` 存储 `[roll,pitch,yaw]`，定义 `R_nb = Rz(yaw) Ry(pitch) Rx(roll)`。提供 Euler→quaternion、quaternion→Euler、quaternion→matrix 转换；矩阵用于输出与坐标变换，不作为积分状态。

姿态配置属于静态拓扑，不在 JIT 内按字符串分支。同一个 batch 使用相同姿态布局；不同布局分别构建和编译。质量、惯量、载荷、资源仍是显式 PyTree 数值参数。

## 方程与载荷

令 `v` 和 `ω` 为机体系速度与角速度，`I` 为质心处惯量张量，`F`、`M` 为不含重力的机体系合力与质心合力矩：

```text
position_dot = R_nb v
v_dot = F/m + R_nb.T g_ned - ω × v
ω_dot = solve(I, M - ω × (I ω))
quaternion_dot = 1/2 quaternion ⊗ [0, ω]
```

欧拉角版本使用完整 3-2-1 运动学映射。非对角惯量受支持；传入的是张量元素本身，注意工程资料中“惯性积”的符号约定可能与张量非对角元素相反。`mass_properties` 在宿主端检查正质量、有限值、对称正定惯量和主惯量三角不等式。直接更新运行时参数的调用者负责维持这些条件。当前没有质量流失或惯量导数项。

重力由 `gravity_ned_m_s2` 单独加入，默认 `[0,0,9.80665]`。上游载荷中不要再次包含重力。`force_at_point(F,r,M)` 将相对质心位置 `r` 处的力转成质心载荷，加入 `r × F`；`sum_loads(...)` 汇总气动、推力等贡献。

方程范围可参照 [MathWorks 固定质量 6DOF 文档](https://www.mathworks.com/help/aeroblks/6dofquaternion.html)。该文档的 DCM 输出为地面→机体，而此处 `rotation_matrix` 返回机体→地面；互接时取转置，并依据旋转定义核对四元数，不能只按字段名复制。这里没有声称与 MATLAB 数值结果逐项对照通过。

## 最小使用

```python
import jax
import jax.numpy as jnp
from aerodrome.models.rigid_body import RigidBody6DoF, BodyLoads, mass_properties

# 应用入口选择精度；库本身不修改全局 JAX 配置。
jax.config.update("jax_enable_x64", True)
body = RigidBody6DoF(attitude="quaternion")
state = body.initialize(
    position_ned_m=[0, 0, -3000],
    velocity_body_m_s=[150, 0, 0],
    euler_rad=jnp.deg2rad(jnp.array([0., 3., 0.])),
)
parameters = mass_properties(1000., jnp.diag(jnp.array([800., 1200., 1500.])))
loads = BodyLoads(jnp.array([1000., 0., -9800.]), jnp.zeros(3))
advance = jax.jit(body.step)
state = advance(state, loads, parameters, .01)
print(jnp.rad2deg(body.euler_angles(state)))
```

此处数字仅演示接口，不代表飞机配平。

`initialize` 是宿主端检查入口，可传 `euler_rad` 或 `quaternion`，不可同时传。批量初始条件先逐个构造再堆叠；已有数值状态可直接通过 JAX PyTree 操作、采样器和 vmap 处理，无需在 JIT 内调用宿主初始化。

## World 与模块组合

```python
from aerodrome.composition import EntitySpec, WorldSpec, build_world
from aerodrome.composition.rigid_body import RigidBodyAssembly

world = build_world(WorldSpec((EntitySpec("aircraft", RigidBodyAssembly(body)),)))
world_state = world.reset(0, {"aircraft": state})
world_parameters = world.parameters({"aircraft": parameters})
following, records = jax.jit(world.step)(world_state, (loads,), world_parameters)
```

默认 `BodyLoads` 在一个 physics tick 内保持，符合外部 MATLAB/采样模块提供离散载荷的语义。外部程序由现有宿主调度器执行，不能放入 JAX 回调。更高精度的纯 JAX 气动反馈使用静态回调：

```python
def load_law(time_s, stage_state, held_inputs, mass_properties, resources):
    # 在这里组合纯函数气动、发动机等载荷，返回 BodyLoads。
    return BodyLoads(-resources["drag"] * stage_state.velocity_body_m_s,
                     jnp.zeros_like(stage_state.omega_body_rad_s))

assembly = RigidBodyAssembly(body, load_fn=load_law)
# 经 world.parameters(..., resources={"drag": ...}) 提供数值资源。
```

RK4 的四个子阶段都会重新调用 `load_law`，不会冻结依赖速度/姿态的力。独立调用 `body.step` 时用闭包捕获静态 `load_fn`，或将其标为 JIT 静态参数。带自身动态状态的发动机/舵机若要求连续强耦合，应在自定义 Assembly 中组合所有状态并联合积分；本接口不会替它们积分隐藏状态。

`RigidBodyAssembly` 是直接 World 接口，尚未额外包装成 ModuleGraph 的向量端口节点。它沿用起始时刻记录约定，现有 Recorder 可选择状态字段。批量与轨迹示例：`python examples/rigid_body.py`，生成 `artifacts/rigid_body/summary.json` 和 `trajectory.npz`。

## 数值边界与验证

- 四元数 RK4 后归一化，保持符号连续，不强制 `w>=0`。它不是长期保结构积分器；步长需按角速度和载荷变化选择。
- 欧拉角在俯仰 ±90° 奇异。宿主初始化拒绝该区域；导数在 `abs(cos(pitch)) <= euler_singularity_cos` 时返回 NaN。检测发生在积分采样阶段，不是连续事件定位，过大的步长可能跨过检测区。大姿态运动使用四元数。
- 四元数转欧拉角在奇异位置选择 roll=0；此显示约定不唯一且不可用于该处的平滑梯度。积分与训练可直接使用四元数/矩阵。
- 当前假设局部平直、非旋转地球、恒定质量惯量；无地球曲率、自转、碰撞约束、柔性或质量交换。
- 测试覆盖 SciPy 旋转约定、解析平动/转动、非对角惯量守恒、Euler/四元数一致性、穿越俯仰奇异点、RK4 子阶段载荷、float32/64、World batch rollout、梯度与 F-16 纵向退化一致性。CPU 已验证；未做 GPU 性能验证。
