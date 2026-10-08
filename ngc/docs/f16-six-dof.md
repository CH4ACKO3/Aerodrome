# F-16 六自由度机体

`models/f16.py` 将完整 ISRL 气动表接入现有四元数刚体。状态为 NED 位置、**相对地面的体轴速度**、body→NED 四元数 `[w,x,y,z]` 和体轴角速度 `[p,q,r]`。统一使用 SI、弧度、FRD 体轴；高度为 `-position_ned_m[2]`。

已实现三轴气动力和力矩、升降舵/副翼/方向舵/前缘襟翼、角速度阻尼、重心修正、惯量积耦合、均匀风场，以及每个 RK4 子阶段重新计算载荷。`Controls` 接收实际舵偏和推力 N；推力沿体轴 +X、通过质心。没有发动机转速、油门映射、舵机滞后或限速。原有纵向模型与 DLQR 实验继续保留。

## 运行

在 `ngc/` 下：

```sh
# 配平直线飞行，记录完整状态；headless 验证姿态投影，不生成画面。
uv run --locked aerodrome --overlay scenario/f16_6dof.yaml --overlay environment/earth.yaml --overlay renderer/headless.yaml runtime.steps=200 renderer.parameters.every_steps=20

# 同一预设也可通过 Hydra 组合。
uv run --locked --extra hydra aerodrome-hydra scenario=f16_6dof environment=earth runtime.steps=200

# 4 秒：配平对照 + 三舵面脉冲，World + JIT + vmap。
# control extra 提供绘图依赖，动力学与配平本身不依赖 python-control。
uv run --locked --extra control python examples/f16_six_dof.py
```

示例产物为 `artifacts/f16_six_dof/{summary.json,trajectory.npz,flight.png}`，可用 `--output` 指定目录。NPZ 第一维区分配平和脉冲两条轨迹，时间轴包含初态与末态。脉冲在 0.5–1.5 s 施加升降舵 −0.25°、副翼 +2°、方向舵 +0.5°，之后回到配平舵量。这是开环响应，没有姿态恢复控制器。

配置项 `surface_offset_rad` 是持续施加的三舵面偏置，顺序为升降舵、副翼、方向舵；`omega_body_rad_s` 是初始角速度；`wind_ned_m_s` 是空气团相对地面的速度。`lef_rad` 在配平和仿真中固定。动态控制序列可通过 World 输入 `Controls`，示例给出完整接线。

## 配平与数值范围

`pipelines/f16_six_dof_trim.trim` 求解 alpha、beta、三舵偏和推力，使全部六个加速度为零。固定 roll=0、theta=alpha、角速度为零；航向可指定。侧滑不强制为零，因为原始数据在 beta=0 处存在小的侧力/滚转/偏航偏置。航向是机头方向，因此与地面航迹角可能不同。

默认 150 m/s、3000 m、g=9.806 m/s²、无风、LEF=0：alpha≈3.5120°，beta≈−0.2693°；升降舵≈−1.37387°、副翼≈0.03824°、方向舵≈−0.62110°，推力≈8398.068 N。加速度残差最大约 `1.78e-15`。4 秒配平对照的高度漂移在本次 float64 CPU 运行中为零。

配平是相对均匀空气团的平飞；水平风改变地速，垂直风会带来等量的地面爬升/下降，不能称为定高度配平。该求解器要求 float64、竖直向下重力；搜索 alpha −5°..15°、beta ±10°、升降舵 ±25°、副翼 ±21.5°、方向舵 ±30°、推力 0..84500 N。推力上界只是求解范围，不是发动机性能曲线。

气动插值共同范围：alpha −20°..45°、beta ±30°、升降舵 ±25°，超域返回 NaN，不截断或外推。副翼/方向舵沿源模型做线性增量叠加，调用方应保持上述舵偏范围；LEF 应为 0°..25°，模型不偷偷限幅。大气复用纵向模型的 NASA Glenn 对流层拟合，适用高度 0..11000 m。查表导数在网格面上有跳变，JAX 自动微分是所在网格单元的局部导数。

## 来源与明确差异

43 张表来自 [ISRL 固定提交](https://github.com/isrlab/F16-Model-Matlab/tree/d019742f2fe7f5f25c521bc7e846519b28b759b3)，MIT 声明保存在 `models/data/f16/LICENSE`。`scripts/import_f16_aerodynamics.py SOURCE_REPO DESTINATION` 按原始列优先布局导出所有表，同轴通道合并查表，不重采样。离线转换需要 h5py，运行时只加载 `aerodynamics.npz`；`aerodynamics.json` 记录数据 hash、轴和通道。

两处建模选择与原 MATLAB 源码不同：

1. `F16AeroFM.m` 的 Cz 俯仰阻尼使用了静态 `delta_Cz_lef`，而已加载的速率表 `delta_Czq_lef` 未被使用。本实现使用后者，和已有纵向模型一致。这是明确的公式修正，不能宣称对原函数逐点完全一致。
2. MATLAB 参数矩阵把 `Ixz=982 slug·ft²` 写成正的非对角项；[Julia 标量转动方程](https://github.com/isrlab/F16Model.jl/blob/8b719078922b8fa786554c2446ba749c4ca7a447/src/NonlinearF16Model.jl)将其用作惯量积。本实现采用惯量张量非对角项 **−982**，与后者方程及归档运动数据一致。其余质量/几何常数沿用来源换算，不混用另一套气动数据。

深失速修正仍为零，发动机陀螺力矩为零，质量/惯量固定；不包含地球曲率、自转、阵风、传感器或飞控系统。

## 验证依据

- 上游 `F16_Julia_Dump.mat` 的最初六个时刻作为只读参考，原样提取到 `tests/data/f16_julia_initial.json`，附源链接和 hash。五阶单边差分反推初始气动力系数和角加速度；六系数绝对误差要求 `<1e-7`，考虑大气密度差异后的角加速度误差 `<2e-6 rad/s²`。初态 p=q=r=0，避免把前述 Czq 修正混入静态对照。
- 所有 43 个查表通道在端点和非节点处与 SciPy 对照；不同 alpha、升降舵、q 和 LEF 下的纵向系数与已有子模型对照。
- 六轴配平、均匀风下的相对运动、舵面导数与有限差分、JIT/vmap、光滑单元内 RK4 四阶收敛。
- 真实 CLI 的配平飞行、状态保存和 headless 姿态投影；真实 World 的双轨迹三舵面脉冲实验。

本轮未启动 MATLAB/Julia；使用的是上游归档输出，不是新跑的跨语言全轨迹验证。运行设备为 CPU，尚未验证全包线、真实发动机/舵机或六自由度闭环控制。

后续已添加[简单姿态 PD 控制例子](f16-attitude-hold.md)，验证局部小扰动的闭环恢复；不改变这里的开环机体实验。
