# 零号工程：组件示例与代码编写

零号工程是教材各章共用的示例库。完整任务入口见[五个实战工程](teaching-projects.md)。章节负责解释原理，在需要运行验证时链接到这里的固定小节；实战工程说明自己组合了哪些组件。小节链接不依赖教材章节编号或显示标题，后续改标题时保留原锚点。

这一页是组件入口，不表示所有列出的扩展都已经实现。各链接中的模型假设和实现状态仍有效；低阶质点示例不能当作完整飞机模型。

| 小节 | 固定链接后缀 | 主要内容 |
|---|---|---|
| 00.1 | `#world` | World、时钟、状态和输入保持 |
| 00.2 | `#coordinates` | 坐标、姿态和单位 |
| 00.3 | `#dynamics` | 运动积分与载荷 |
| 00.4 | `#fixed-wing` | 固定翼、配平和局部模型 |
| 00.5 | `#rotorcraft` | 旋翼机模型的计划与边界 |
| 00.6 | `#estimation` | 观测、噪声与状态估计 |
| 00.7 | `#control` | 控制算法 |
| 00.8 | `#reference` | 参考信号与跟踪 |
| 00.9 | `#environment` | 环境、障碍和事件 |
| 00.10 | `#learning` | 数据、学习与策略接口 |
| 00.11 | `#evaluation` | 批量实验与公平比较 |
| 00.12 | `#replay` | 结果记录与回放 |
| 00.13 | `#coding-styles` | 脚本式、函数式、面向对象式 |
| 00.14 | `#math-tools` | 测量、拟合、决策与有限时域倒推 |
| 00.15 | `#statistical-learning` | 分类、回归、表示与学习评测短例 |
| 00.17 | `#symbolic-state-conversion` | 状态换元、导数求解与符号残差检查 |

<a id="world"></a>

## 00.1 World、时钟与状态

阅读 [World](world.md) 与 [统一配置](configuration.md)，运行其中的基础例子。观察控制周期与物理积分步长的区别，以及一条输入如何在多个 tick 之间保持。数学小项目可以直接调用数值核；需要多速率和实体组合时再接 World，不要求每个短公式都套入 World。

<a id="coordinates"></a>

## 00.2 坐标、姿态与单位

入口为 [仿真基础工具](simulation-tools.md) 和 [地理与大气](geography-atmosphere.md)。重点分清地面速度/空速、NED/FRD、角度单位和四元数方向；用数值坐标变换验证图像直觉。

<a id="dynamics"></a>

## 00.3 运动积分与载荷

完整刚体入口为 [六自由度刚体](rigid-body.md)。数学工程还共用 `aerodrome.models.point_mass`：`PointMassState(position_m, velocity_m_s)` 和 `step(state, acceleration_m_s2, dt_s)`。输入是步内保持的实际加速度，精确更新位置与速度；支持 NumPy/JAX 和批次形状，不包含机翼、旋翼或重力假设。位置必须使用旧速度加半个加速度项，不应误用新速度造成额外位移。

<a id="fixed-wing"></a>

## 00.4 固定翼与配平

从 [F-16 六自由度机体](f16-six-dof.md) 的平飞和舵面脉冲开始，再看 [纵向配平与 LQR](f16-level-flight.md)。区分完整机体、纵向简化模型和指定工况的局部线性模型。

<a id="rotorcraft"></a>

## 00.5 旋翼机模型

完整四旋翼动力学与电机/姿态控制仍待实现。现有可停止的质点用于理解运动规划和队形规则，不具备旋翼混控、倾斜产生水平加速度或旋翼失效等物理含义。后续旋翼机示例继续挂在这个锚点下，不另造教材中的第二套模型。

<a id="estimation"></a>

## 00.6 观测与估计

入口为 [基础闭环方程](equations.md)、[传感器与记录](telemetry.md)。现有俯仰 KF 是一个具体的二状态例子；[工程 05 的效能辨识](../projects/adaptive_control/README.md)是另一个估计问题，不将它称为完整导航系统。练习时明确哪一条是控制器可见观测，哪一条真值只用于评分。

<a id="control"></a>

## 00.7 控制算法

从教程站的 [速度控制实验](/Aerodrome/experiments/velocity-control/)进入比例反馈，再运行 [F-16 姿态 PD](f16-attitude-hold.md)。通过参考误差、实际控制量和状态响应一起判断控制效果；姿态稳定不等于高度或航向保持。

<a id="reference"></a>

## 00.8 参考与跟踪

比较固定目标、连续参考和突变参考时的响应。入口包括 [线性系统](linear-control.md) 与[工程 02 的动态参考跟踪](../projects/dynamic_decision/README.md)。参考是否预先可知、可预览多少步，都是实验条件，不能在比较两种算法时暗中改变。

<a id="environment"></a>

## 00.9 环境与事件

入口为 [环境物理量](geography-atmosphere.md) 与[工程 01 的几何障碍](../projects/obstacle_navigation/README.md)和[工程 03 的成员事件](../projects/variable_team/README.md)。到达、碰撞、加入/退出、故障等事件都记录发生时间；记录帧之间的运动也可能碰撞，不能只靠漂亮的离散轨迹图判定成功。

<a id="learning"></a>

## 00.10 数据与学习接口

阅读 [Gymnasium 接线](gymnasium.md)，再看[工程 04](../projects/task_scheduling/README.md)如何生成处理时间数据、拟合回归模型及运行调度。框架适配器不等于训练算法。先用能解释的线性回归、局部拟合或表格模型展示完整数据流程，再决定是否需要神经网络。

<a id="evaluation"></a>

## 00.11 批量与评测

入口为 [批量运行](batch.md)。训练、验证、测试分开；地图、规模、参考信号和故障类型的留出各有不同含义。结果保留失败和最差案例，不能只比较最漂亮的一条曲线。静态匹配模型与动态调度模型也要分别评价。

<a id="replay"></a>

## 00.12 记录与回放

阅读 [渲染数据](rendering.md) 并打开 [场景回放](/Aerodrome/experiments/scene-viewer/)。记录中的时间是仿真时间；展示下采样、倍速播放不会改变实际积分。网页参数修改后未重算时，旧轨迹仍属于旧配置。

<a id="coding-styles"></a>

## 00.13 代码编写：三种组织方式

同一个速度闭环用三种写法实现：脚本从上到下执行；函数明确接收参数并返回结果；对象保存当前状态并提供 reset/step/run。三个程序共享物理更新，数值结果保持一致。

完整课程、运行命令和练习见 [代码编写教学](code-styles.md)。这三种方式可以混合使用，不是从“低级”到“高级”的排名；选择取决于实验是否需要复用、批量计算或状态生命周期。

<a id="math-tools"></a>

## 00.14 数学工具：从手算到程序

[数学工具算例](math-tools.md)提供两个独立短程序。第一个串起传感器融合、带噪位置拟合、重复实验、正则化与任务分配；第二个运行三状态、两步的策略评价和动态规划。每个程序都能直接对照教材的手算数值，并通过少量参数改变实验条件。

[贝叶斯更新与抽样](bayesian-update.md)补充 Beta 和高斯共轭更新，比较同一分布的不同大小样本集。

这些程序只依赖 NumPy，使用教学设定的数据和已知模型，不需要启动完整仿真器。它们用于理解数学运算；控制器、机体和传感器的实际接线仍见前面的组件小节。

## 章节作者如何链接

章节示例使用稳定路径，例如：

```md
[运行组件示例：三种代码写法](/Aerodrome/reference/project-zero/#coding-styles)
[运行组件示例：姿态控制](/Aerodrome/reference/project-zero/#control)
```

只在该章实际出现相关示例时添加链接；尚未写好的章节不机械堆满入口。链接可以直接到小节，也可继续进入详细实验页。新增组件小节时先在此登记固定锚点，重命名标题时保留锚点。


<a id="statistical-learning"></a>

## 00.15 统计学习算例

[统计学习算例](statistical-learning.md)提供第二章各方法主题对应的十五个 NumPy 短例。从生成式分类、逻辑回归和最小二乘，逐步到卷积、注意力、近邻、GP、树、PCA、聚类、矩阵补全和图聚合。逻辑回归例包含训练、验证与测试流程；其他例子按各节需要展示拟合或给定参数下的运算，文档分别说明。

先在短例中核对输入、输出、假设与评价指标，再到[数据与学习接口](#learning)对照完整任务的数据生成和方法比较。教学中的前向计算不等于已经训练出可投入运动控制的模型。

第二章现按五个主题组织。统计学习算例之外，还可运行[采样 Q 学习](statistical-learning.md#q-learning)与[比例控制参数搜索](statistical-learning.md#parametric-control)，分别观察长期代价估计和闭环参数选择。两者使用同一个短程序 `examples/learning_control.py`，无需完整机体仿真环境。

[神经网络的完整前向与反向计算](statistical-learning.md#mlp-backprop)用 `examples/mlp_backprop.py` 逐项显示线性层、激活、损失、外积梯度和一次参数更新，并通过中心差分核对梯度，便于把矩阵求导与实际数组对应起来。


<a id="reinforcement-learning"></a>

## 00.16 动态规划与强化学习

[第三章算例入口](reinforcement-learning.md)连接动态规划、最优反馈、预测控制与策略参数优化。连续控制例使用速度积分器，离散例使用小型状态空间；先核对模型、时域和代价，再比较算法，不能把不同任务的总代价放进同一排名。

[动态规划与价值学习程序](rl-dp-control.md)解释精确解、在线预测与采样更新；[策略优化程序](rl-policy-learning.md)解释参数改变怎样影响整条轨迹。两个程序只依赖 NumPy，运行方式和实际覆盖范围分别记录在对应说明中。它们是低阶教学模型，完整机体接线继续使用前面的动力学、控制与评测组件。


<a id="symbolic-state-conversion"></a>

## 00.17 动力学方程的符号转换

[4.3.7 的等价变换推导](/Aerodrome/chapters/04-aircraft-control-models/03-rigid-body-dynamics/#faq-dynamics-to-simulation)对应 `examples/symbolic_state_conversion.py`。程序接收已声明的 SymPy 符号、方程残差或状态映射，求解对导数为仿射的方阵系统，返回导数表达式、成立条件和代回残差。直角坐标到极坐标的例子在位置 $(3,4)$、速度 $(2,-1)$ 处得到 $\dot r=0.4$、$\dot\varphi=-0.44$；移动原点的例子核对显式时间偏导。

在 `ngc/` 目录运行：

```sh title="符号状态转换"
uv run --no-project --with sympy python examples/symbolic_state_conversion.py
```

也可[下载独立脚本](/Aerodrome/downloads/symbolic_state_conversion.py)，在其所在目录运行 `uv run --no-project --with sympy python symbolic_state_conversion.py`。输出为 SymPy 对象字典；变量的单位、坐标方向、角度分支和原方程定义域在转写模型时明确。
