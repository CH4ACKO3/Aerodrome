# F-16 纵向平飞与扰动恢复

本实验从公开 F-16 气动表构造**对称纵向非线性子模型**，完成配平、JAX 自动微分线性化、离散 LQR 设计和非线性闭环验证。它不是此前的假设系数二状态模型，也不等同完整六自由度 F-16 或实机飞控。

## 来源与运行

来源：[ISRL F16-Model-Matlab](https://github.com/isrlab/F16-Model-Matlab/tree/d019742f2fe7f5f25c521bc7e846519b28b759b3)，提交 `d019742f2fe7f5f25c521bc7e846519b28b759b3`，NASA TP-1538 数据家族。移植参照 `F16AeroFM.m` 与 `load_F16_params.m` 的纵向部分。MIT 声明和数据 manifest 保存在 `src/aerodrome/models/data/f16/`。

原 HDF5 SHA-256：`b9ac8d21cfb749c0e9897766d5f7b3cdcfdafc86ef8e948c064aafdc2e453f07`；抽取 NPZ：`8c59021955eed05cf3ee41f99fd9d8df634558684f018daebe0f2795f09dac4b`。转换复现 MATLAB 列主序 reshape，取 beta=0 的已有节点截面，没有拟合/重采样。运行时校验 NPZ 哈希。

```sh
uv run --locked --extra control python examples/f16_level_flight.py
```

结果保存在 `artifacts/f16_level_flight/`：`summary.json`、`trajectory.npz`、`flight.png`。NPZ 含 120 秒轨迹、控制量、配平、线性矩阵和增益；图显示前 40 秒。蓝色为配平初态，橙色/绿色分别为正/负扰动。

离线重建使用 `scripts/import_f16_longitudinal.py SOURCE_REPO DESTINATION`，需要 h5py。本次转换在旧 Windows Python 3.12 工具环境使用 h5py 3.16.0 完成；实际仿真仍是 Python 3.14t WSL，运行路径不加载 h5py。

## 方程与边界

`x=[V,alpha,q,theta,h]` 分别为空速 m/s、迎角 rad、俯仰角速度 rad/s、俯仰角 rad、高度 m。输入 `[elevator_rad,thrust_N]`。机体系 z 向下，高度向上；侧滑、滚转和偏航运动约束为零。

保留基本 Cx/Cz/Cm、q 阻尼、前缘襟翼修正、升降舵效率、重心修正和 deltaCm。角度以度查表，域外返回 NaN，不静默外推。与上游一样忽略 deep-stall 修正。LEF 固定 0°，相应气动修正仍按原公式计算，并非直接删除 LEF 修正。

令 `X=qbar*S*Cx+T`、`Z=qbar*S*Cz`、`M=qbar*S*cbar*Cm`、`gamma=theta-alpha`：

\[
\dot V=(X\cos\alpha+Z\sin\alpha)/m-g\sin\gamma,
\quad\dot\alpha=q+(Z\cos\alpha-X\sin\alpha)/(mV)+g\cos\gamma/V
\]
\[
\dot q=M/I_{yy},\quad\dot\theta=q,\quad\dot h=V\sin\gamma.
\]

质量、惯量、重心和几何沿用源文件 SI 转换。大气使用 [NASA Glenn 米制对流层拟合](https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html)，只开放 0–11000 m，无风；不声称复现原 Simulink 的全部环境模块。

## 配平与控制器

V=150 m/s、h=3000 m，令 q=0、theta=alpha，求 Vdot=alphadot=qdot=0：

| 配平量 | 结果 |
|---|---:|
| 迎角 / 俯仰角 | 约 3.506° |
| 升降舵 | -1.377779° |
| 推力 | 8401.448 N |
| 最大配平导数绝对值 | 4.90e-16，各分量按自身 SI 单位 |

局部 A/B 用 JAX 自动微分得到，经状态/输入归一化，以现有 python-control 适配器作 0.02 s ZOH 离散化并设计 DLQR。归一化尺度为 `[5 m/s,2°,2°/s,2°,20 m]` 和 `[5°,10000 N]`；Q=diag(2,1,0.2,0.5,2)，R=diag(25,2)，用较高升降舵代价降低控制突变。

控制律 `u=u_trim-K*(observation-x_trim)` 恢复同一配平点，并非通用航迹/速度指令控制器。本轮明确使用理想全状态导航；理想执行器仅有幅值限制：升降舵 ±25°、推力 0–19000 lbf 对应的 N。没有舵机惯性、速率限制或发动机动态。零推力下限是本实验设置，不是源配平脚本或真实发动机怠速条件。

物理 RK4 dt=0.01 s，控制每 2 tick 更新并保持。3 个 World 分别从配平、正扰动、负扰动初态开始，扰动为 `±[5 m/s,1°,0.5°/s,2°,30 m]`。

## 结果与验证

- 配平 World 在 120 s 内保持平飞。
- 正/负扰动在 **20.34 s / 20.66 s** 后同时进入并持续保持全部误差限：速度 0.1 m/s、迎角/俯仰角 0.05°、俯仰角速度 0.02°/s、高度 0.2 m。
- 最大高度偏差为 32.71 m / 32.67 m；最大俯仰角速度约 5.18°/s / 4.99°/s，暂态有超调。
- 正扰动的理想推力触及零下限，共 129 个决策步限幅；负扰动无幅值限幅。
- 离散线性闭环最大极点模 0.995814，稳定性结论为局部。极小最终误差来自无噪声、精确模型和固定目标，不代表实机精度。

自动测试覆盖数据完整性、源节点、插值与 SciPy 对照、域外处理、体轴方程独立投影、配平、雅可比有限差分、Riccati 残差、非线性恢复和步长减半对照。尚未运行 MATLAB/Julia 六自由度参考，因此不是跨实现认证。

下一步增加发动机/舵机动态与速率限制，再接传感器和估计器，最后验证质量/重心变化、风扰动及多配平点。当前保留为可复现的理想闭环基线。
