# 仿真基础工具

公共工具放在 `aerodrome.core` 下，供物理、传感器、导航、制导和控制模块共同使用。数值函数使用 JAX，不依赖 SciPy、文件读取或宿主回调；不修改全局精度设置。向量和姿态函数处理单个对象，批量调用使用 `jax.vmap`，时间循环使用现有 `scan`。标量单位函数可直接处理数组。

## 统一约定

- 内部 SI 和弧度；向量 `(3,)`、矩阵 `(3,3)`，按列向量理解。
- `R_ab` 表示 **b 系坐标到 a 系坐标**，即 `v_a = R_ab @ v_b`。函数不会检查数值矩阵的正交性，调用者须提供 proper rotation。
- NED 为北、东、下，机体系 FRD 为前、右、下。四元数为 Hamilton `[w,x,y,z]`，欧拉角按 `[roll,pitch,yaw]` 存放。
- 原点不同的位置变换必须包含平移；速度、力等自由向量只旋转。角速度映射成欧拉角导数也不是简单旋转。

## 已提供的工具

| 模块 | 功能 |
|---|---|
| `core.rotations` | 欧拉角/四元数/旋转矩阵、旋转向量→四元数、乘法/共轭/逆、姿态误差、向量旋转、叉乘矩阵、欧拉角导数↔机体角速度 |
| `core.frames` | 刚性坐标变换及组合/求逆、点与向量、协方差、力/力矩、传感器安装偏置处的速度与加速度 |
| `core.frames` | NED↔ENU、FRD↔FLU、机体↔NED、ECEF→局部 NED 矩阵和位置变换 |
| `core.airdata` | 风速扣除、空速/迎角/侧滑角及反向构造、机体↔风轴、升阻力变换、动压、航迹方位角和爬升角 |
| `core.units` | 度↔弧度、英尺↔米、节↔米每秒、力和质量换算常数、角度归一化及最短角差 |

原 `models.rigid_body` 的姿态函数现在复用公共实现，旧导入仍兼容。没有改变 F-16 来源模型里的历史单位换算常数，以免静默改变来源复现结果。

## 飞机状态到气动力

```python
import jax
import jax.numpy as jnp
from aerodrome.core.rotations import euler321_to_matrix
from aerodrome.core.airdata import (
    air_relative_velocity, airdata, aerodynamic_force_body, dynamic_pressure,
)

def evaluate(euler, velocity_body, wind_ned, density, area, cd, cl):
    R_nb = euler321_to_matrix(euler)
    relative = air_relative_velocity(velocity_body, wind_ned, R_nb)
    data = airdata(relative)
    scale = dynamic_pressure(density, data.speed_m_s) * area
    force = aerodynamic_force_body(scale*cd, 0., scale*cl,
                                   data.alpha_rad, data.beta_rad)
    return force, data

compiled = jax.jit(evaluate)
force, data = compiled(jnp.zeros(3), jnp.array([100., 0., 5.]),
                       jnp.array([10., 0., 0.]), 1.2, 20., .03, .4)
# force 可交给 BodyLoads；气动力矩仍由气动模型提供。
```

`wind_ned` 是空气团相对地面的速度，不能直接传气象“来向”。风轴 x 沿飞机相对空气的速度，风轴力采用 `[-D, Y, -L]`。零/过低空速或纯侧向速度时，完整气动角描述无效：`angles_valid=False`，两个角为 NaN，不伪造零迎角。纯侧向时侧滑本身可定义，但迎角及完整风轴方向不唯一，因此这里统一拒绝整组角。调用者应依据模型有效域终止、切换模型或实施显式低速规则。该区和分支边界不保证梯度；普通有效域已测自动微分。

## 位置、载荷与传感器安装点

```python
from aerodrome.core.frames import (
    Transform, transform_point, transform_vector, transform_wrench,
    inverse_transform, compose_transforms, point_velocity, point_acceleration,
)

# B 原点在 A 系中的坐标为 translation；rotation 为 R_ab。
a_from_b = Transform(R_ab, translation)
point_a = transform_point(a_from_b, point_b)
vector_a = transform_vector(R_ab, vector_b)
force_a, moment_about_a = transform_wrench(a_from_b, force_b, moment_about_b)
```

载荷变换加入 `translation × force_a`，得到关于 A 原点的力矩。不要把这个接口用于只换轴、仍关于同一个物理参考点的力矩而误加平移项。`transform_covariance` 仅变换三维向量协方差 `R P R.T`，不是完整误差状态协方差或 SE(3) adjoint。

`point_velocity(v, omega, offset)` 与 `point_acceleration(a, omega, omega_dot, offset)` 的全部输入须在同一坐标系，offset 为固连刚体的安装偏置。这些工具用于杆臂效应；不包含传感器特定力、重力扣除或相对运动产生的科氏项。

## 地理坐标与姿态误差边界

`ecef_to_ned_matrix(latitude_rad, longitude_rad)` 使用**大地纬度**。位置变换额外提供参考点 `origin_ecef_m`，由调用者确保它与参考经纬度一致。逆向位置变换使用 `inverse_transform`。此文件中的 frames 接口只建立局部坐标基；经纬高↔ECEF 椭球求解现由 core.geodesy 提供，见 [地理位置与空气物理量](geography-atmosphere.md)。地球自转动力学和 ECI 时间变换仍未实现；高精度地球尺度定位宜使用 float64。

`quaternion_error(reference,current)` 返回父坐标系中的最短旋转误差，满足 `R_error = R_reference @ R_current.T`，不是三维角度残差。乘法右侧先作用；四元数逆支持非单位非零输入。`rotation_vector_to_quaternion` 在零旋转处使用 Taylor 分支，值和一阶导数有限。欧拉角仍具有俯仰奇异性；最短误差在 180°、角度 wrap 在分界点不连续，勿在那里假设可微。

## 验证与来源

测试见 `tests/modules/test_simulation_tools.py`：SciPy 独立旋转对照、组合与逆、零旋转 Jacobian、旋转矩阵导数验证欧拉角速度映射、解析力矩/杆臂、赤道基向量、风轴对齐与升力符号、速度与气动角往返、有限差分梯度、单位及 ±π 边界。旧刚体和 F-16 测试也参与完整回归。

局部基方向参考 [ESA Navipedia ECEF/ENU 变换](https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)，随后按 NED/ENU 轴定义转换；风轴定义参照 [MathWorks body-to-wind DCM](https://www.mathworks.com/help/aerotbx/ug/dcmbody2wind.html)。没有执行 MATLAB 对照程序。
