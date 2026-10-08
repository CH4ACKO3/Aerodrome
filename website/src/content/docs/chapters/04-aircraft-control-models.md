---
title: "4.飞行器建模与数值仿真"
description: 从运动和载荷出发，建立可计算的飞行器模型，推进时间，分析配平与局部运动，并验证仿真结果。
---

让飞机抬高一点机头，升力、速度和高度都会跟着变化；让四旋翼的一对电机加速，推力与姿态也会一起改变。前一章已经讨论怎样选择控制输入，现在需要把输入接到一架能在计算机中运动的飞行器上：给定它当前的位置、速度和姿态，算出受到的力与力矩，再求下一时刻的运动。

这条计算链有几个相互联系的环节。坐标与姿态决定一股力朝向哪里，动力学决定这股力使状态怎样变化，数值积分决定如何沿时间推进；气动、推进、环境和执行器则给出具体载荷。把它们接起来，就能解释轨迹，寻找稳定飞行条件，并为导航、控制和学习提供模型与观测。

本章沿着“描述运动 → 计算载荷与导数 → 推进时间 → 组装机型 → 配平和验证”展开。主线使用局部平直地面、恒定质量与惯量的刚体，统一采用 SI 单位、弧度、NED 地面轴和 FRD 机体轴。

## 从一次操纵输入开始

[4.1 从真实飞行器到仿真模型](/Aerodrome/chapters/04-aircraft-control-models/01-simulation-models/)从舵面和电机输入出发，说明状态、输入、参数和观测各自记录什么。质点、纵向模型与六自由度模型保留不同的运动细节，选择依据是希望解释的飞行现象。单步流程把配置、模型、积分器和记录连接起来，让后面的公式都有明确的计算位置。

## 用坐标与姿态描述空间运动

[4.2 坐标系与姿态表示](/Aerodrome/chapters/04-aircraft-control-models/02-frames-and-attitude/)将地面位置、机体速度和风轴载荷放到同一套约定中。通过旋转顺序示意和一次 90° 偏航手算，理解欧拉角、旋转矩阵与四元数的对应关系，再由角速度求姿态变化率。这些关系也用于后续章节中的传感器安装和导航计算。

## 由合力与合力矩得到状态导数

[4.3 六自由度刚体动力学](/Aerodrome/chapters/04-aircraft-control-models/03-rigid-body-dynamics/)从 Newton–Euler 方程推导位置、速度、姿态与角速度四组导数。转动坐标系中的交叉项、惯量耦合和偏置力的力矩在这里获得具体含义。自由落体与无外力矩转动提供可以直接核对的答案，同一刚体接口随后接收固定翼或旋翼产生的载荷。

## 沿时间推进模型

[4.4 数值积分与离散仿真](/Aerodrome/chapters/04-aircraft-control-models/04-numerical-simulation/)比较 Euler、二阶方法与 RK4，解释步长如何影响误差和稳定性。姿态归一化、子阶段载荷计算、输入保持与多速率采样共同决定实际运行时序。通过减小步长观察误差变化，可以逐步判断轨迹中的偏差来自数值计算还是物理模型。

## 加入空气、部件和测量

[4.5 环境、执行器与测量模型](/Aerodrome/chapters/04-aircraft-control-models/05-environment-and-components/)把地速变成相对空气的速度，计算气动所需的大气量，并用简单动态描述舵机和电机对指令的响应。传感器再从状态生成有采样、噪声、偏置或延迟的测量。这一步将模拟器内部的运动真值与控制器实际使用的信息联系起来。

## 组装固定翼与旋翼模型

[4.6 固定翼飞行器建模](/Aerodrome/chapters/04-aircraft-control-models/06-fixed-wing-models/)由动压、参考面积、气动力系数和力矩系数计算载荷，解释迎角、侧滑、舵偏与角速度怎样进入模型。教学参数用于手算，现有 F-16 查表模型用于观察完整机体的平飞与开环响应，纵向简化则保留其中最常用的一条分析路线。

[4.7 旋翼与多旋翼飞行器建模](/Aerodrome/chapters/04-aircraft-control-models/07-rotorcraft-models/)从单个旋翼的推力和反扭矩出发，把位置杆臂、旋向和各电机转速合成整机载荷。四旋翼悬停与差动输入是主要算例，倾斜推力解释水平加速度的来源；直升机的总距、周期变距、尾桨和挥舞说明另一类旋翼布局怎样产生控制作用。

## 找到飞行条件，再解释附近的运动

[4.8 配平、线性化与运动分析](/Aerodrome/chapters/04-aircraft-control-models/08-trim-and-linearization/)寻找平飞、转弯和悬停所需的状态与输入，再研究小扰动怎样演化。配平残差、局部三维姿态误差、雅可比和运动模态，将非线性机体接回第一章的矩阵工具与第三章的控制模型。位置随时间变化的稳定飞行也能作为分析参照。

## 用可核对的实验检验仿真

[4.9 仿真验证与工程实验](/Aerodrome/chapters/04-aircraft-control-models/09-validation-and-experiments/)从解析运动和步长收敛出发，逐步加入独立参考、实测数据与参数不确定性。记录时间戳、配置和输入序列，使轨迹能够重现；批量实验与软件、硬件在环则把单次正确计算扩展到更接近实际运行条件的验证。

## 常见问题

- [向量怎样换到另一个坐标系？转换矩阵、欧拉角奇异性与 SVD 怎样处理？](/Aerodrome/chapters/04-aircraft-control-models/02-frames-and-attitude/#faq-frame-transforms)
- [不同形式的动力学方程何时等价，怎样用符号程序完成转换？](/Aerodrome/chapters/04-aircraft-control-models/03-rigid-body-dynamics/#faq-dynamics-to-simulation)

## 阅读路线与三条贯穿算例

初读依次完成 4.1–4.5，再根据机型进入固定翼或旋翼部分，最后用 4.8、4.9 分析与验证。第一条算例是自由刚体：载荷简单，便于核对坐标、动力学与积分。第二条是固定翼平飞：从气动计算进入配平、线性化与输入扰动。第三条是四旋翼悬停：由推力平衡进入电机差动、姿态变化和倾斜加速。

程序入口集中在[零号工程](/Aerodrome/reference/project-zero/)：先运行[通用刚体](/Aerodrome/reference/project-zero/#dynamics)，再进入[固定翼与配平](/Aerodrome/reference/project-zero/#fixed-wing)；[旋翼入口](/Aerodrome/reference/project-zero/#rotorcraft)记录当前组件状态，4.7 的教学算例由给定方程核算。模型建立后，可回到[第三章](/Aerodrome/chapters/03-reinforcement-learning/)比较控制方法；导航与飞行控制的专题分别由后续章节承接。

主要参考为 Stevens、Lewis、Johnson 的 [*Aircraft Control and Simulation*，第三版](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119174882)。各小节按本章的计算主线组织，列出对应原书节号和印刷页码，方便继续阅读。
