---
title: "1.1 线性代数与矩阵计算"
description: 从状态与测量理解线性映射、矩阵分解，以及形状明确的向量和矩阵求导。
tableOfContents:
  minHeadingLevel: 2
  maxHeadingLevel: 3
---

一架飞机的状态通常不能用一个数描述。位置、速度、姿态和角速度组成状态；传感器提供测量；控制器把状态误差转成舵面和推力指令。线性代数让我们把这些量之间的关系写清楚，矩阵求导进一步回答：**状态或参数改变一点，预测、误差和控制效果会改变多少？**

本节以 Murphy 的 *Probabilistic Machine Learning: An Introduction* 第 7 章为主要参考，按手册后续需要重新组织，展开求导规则，并另行编写工程算例。阅读只要求熟悉一元函数求导、求和符号及基础三角函数。概率分布和优化算法在后续页面介绍，这里先准备它们使用的工具。

## 1.1.1 从状态与测量认识向量

考虑沿直线运动的简化飞行，把位置 $p$ 和速度 $v$ 放在一起：

$$
\boldsymbol{x}=\begin{bmatrix}p\\v\end{bmatrix}\in\mathbb R^2.
$$

向量可以表示几何方向，也可以只是按照明确顺序排列的状态分量。上面两个分量分别用米和米每秒计量，不能直接当成平面上的物理箭头。**数组形状相同，不代表物理含义、单位和坐标系相同。**

在时间间隔 $\Delta t$ 内假设速度不变，则

$$
\boldsymbol{x}_{k+1}=\boldsymbol{A}\boldsymbol{x}_k,\qquad
\boldsymbol{A}=\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix}.
$$

矩阵中右上角的 $\Delta t$ 有时间单位，使 $\Delta t\,v$ 与 $p$ 可以相加。矩阵运算的检查包括两件事：行列数能否相乘，以及相加的物理量是否具有一致单位。

本节及后续页面采用以下约定。

| 对象 | 记号与形状 | 约定 |
|---|---|---|
| 标量 | $a\in\mathbb R$ | 不加粗 |
| 列向量 | $\boldsymbol{x}\in\mathbb R^n$ | 数学上视为 $n\times1$ |
| 行向量 | $\boldsymbol{x}^{\mathsf T}\in\mathbb R^{1\times n}$ | 明确写出转置 |
| 矩阵 | $\boldsymbol{A}\in\mathbb R^{m\times n}$ | 第一个维度是行数 |
| 元素 | $A_{ij}$ | 第 $i$ 行、第 $j$ 列；数学下标从 1 开始 |
| 单位矩阵 | $\boldsymbol{I}_n$ | 对角元素为 1，其余为 0 |
| 零向量或零矩阵 | $\boldsymbol{0}$ | 形状由上下文注明 |

除特别说明，我们讨论实数、普通欧氏内积以及可微函数。复数共轭求导和旋转流形上的求导暂不展开。

### 一批记录怎样排列

若有 $N$ 个样本，每个样本有 $d$ 个特征，本手册默认每行一个样本：

$$
\boldsymbol{X}=\begin{bmatrix}\boldsymbol{x}_1^{\mathsf T}\\\vdots\\\boldsymbol{x}_N^{\mathsf T}\end{bmatrix}
\in\mathbb R^{N\times d}.
$$

单个样本在数学上仍是列向量，存入数据矩阵时转成一行。因此，单样本模型 $\boldsymbol{y}=\boldsymbol{A}\boldsymbol{x}$ 的整批计算写成 $\boldsymbol{Y}=\boldsymbol{X}\boldsymbol{A}^{\mathsf T}$。转置来自存储约定，不是另一套物理模型。

## 1.1.2 线性组合、基与矩阵乘法

设 $\boldsymbol{A}=[\boldsymbol{a}_1\ \cdots\ \boldsymbol{a}_n]$，则

$$
\boldsymbol{A}\boldsymbol{x}=\sum_{j=1}^n x_j\boldsymbol{a}_j.
$$

矩阵乘向量，就是按各分量对矩阵的列加权。所有可能结果组成矩阵的**列空间**，也就是这些列能够张成的空间。

若一组向量只有在所有系数为零时，线性组合才等于零，称它们线性无关。能张成一个空间的一组线性无关向量称为该空间的一组基。矩阵的秩 $\operatorname{rank}(\boldsymbol{A})$ 是列空间的维数，也等于行空间的维数。

零空间包含所有被映射成零的输入：

$$
\ker\boldsymbol{A}=\{\boldsymbol{z}:\boldsymbol{A}\boldsymbol{z}=\boldsymbol{0}\},\qquad
\dim\ker\boldsymbol{A}+\operatorname{rank}(\boldsymbol{A})=n.
$$

例如测量 $y=[1\ 0]\boldsymbol{x}=p$ 无法区分 $[p,v]^{\mathsf T}$ 与 $[p,v+\delta v]^{\mathsf T}$。速度方向属于这一次测量的零空间。不过，结合多时刻位置测量和运动模型仍可能推断速度；单个测量矩阵的秩不能替代动态系统的可观测性分析。

### 三种不同的乘法

| 运算 | 形状 | 含义 |
|---|---|---|
| 内积 $\boldsymbol{a}^{\mathsf T}\boldsymbol{b}$ | $(1\times n)(n\times1)\to1$ | 对应分量乘积求和 |
| 外积 $\boldsymbol{a}\boldsymbol{b}^{\mathsf T}$ | $(m\times1)(1\times n)\to m\times n$ | 第 $(i,j)$ 项为 $a_i b_j$ |
| 逐元素积 $\boldsymbol{A}\odot\boldsymbol{B}$ | 两者同形状 | 对应元素相乘 |

矩阵乘法表示映射复合。若 $\boldsymbol{y}=\boldsymbol{B}\boldsymbol{x}$、$\boldsymbol{z}=\boldsymbol{A}\boldsymbol{y}$，则 $\boldsymbol{z}=\boldsymbol{A}\boldsymbol{B}\boldsymbol{x}$，右边的映射先作用。一般不能交换次序；转置时必须倒序：

$$
(\boldsymbol{A}\boldsymbol{B})^{\mathsf T}=\boldsymbol{B}^{\mathsf T}\boldsymbol{A}^{\mathsf T}.
$$

同一几何向量更换基后，坐标也会改变。若新基各列用旧坐标表示为可逆矩阵 $\boldsymbol{B}$，则 $\boldsymbol{x}_{\rm old}=\boldsymbol{B}\boldsymbol{x}_{\rm new}$。当基正交归一时，$\boldsymbol{B}^{-1}=\boldsymbol{B}^{\mathsf T}$。正交矩阵保持长度和夹角，但可能包含反射；三维纯旋转还要求行列式为 $+1$。

## 1.1.3 误差的大小：范数与二次型

三个常用向量范数为

$$
\|\boldsymbol{x}\|_1=\sum_i|x_i|,\qquad
\|\boldsymbol{x}\|_2=\sqrt{\sum_i x_i^2},\qquad
\|\boldsymbol{x}\|_\infty=\max_i|x_i|.
$$

它们分别强调总绝对偏差、欧氏长度和最大分量。矩阵的 Frobenius 范数把所有元素看成一个长向量：

$$
\|\boldsymbol{A}\|_F^2=\sum_{i,j}A_{ij}^2=\operatorname{tr}(\boldsymbol{A}^{\mathsf T}\boldsymbol{A}).
$$

$\operatorname{tr}$ 是方阵对角元素之和。另一个常用量是谱范数 $\|\boldsymbol{A}\|_2=\max_{\|\boldsymbol{x}\|_2=1}\|\boldsymbol{A}\boldsymbol{x}\|_2$，表示映射对长度的最大放大倍数，与 Frobenius 范数通常不同。

工程中不同误差的容许范围不同。若位置、速度的误差尺度分别为 $s_p,s_v>0$，先无量纲化再评价：

$$
L=\frac12\left[\left(\frac{\delta p}{s_p}\right)^2+\left(\frac{\delta v}{s_v}\right)^2\right]
=\frac12\boldsymbol{e}^{\mathsf T}\boldsymbol{W}\boldsymbol{e},\qquad
\boldsymbol{W}=\operatorname{diag}(s_p^{-2},s_v^{-2}).
$$

直接相加位置误差平方与速度误差平方既混合单位，也隐含了随单位选择而变化的权重。状态缩放会影响数值条件和梯度方向，不能只当成显示格式处理。

对称矩阵 $\boldsymbol{W}$ 若满足任意非零 $\boldsymbol{e}$ 都有 $\boldsymbol{e}^{\mathsf T}\boldsymbol{W}\boldsymbol{e}>0$，称为正定，记作 $\boldsymbol{W}\succ0$；若允许等于零，称为半正定，记作 $\boldsymbol{W}\succeq0$。正定权重惩罚所有非零误差；半正定权重可能忽略某些方向。

协方差 $\boldsymbol{\Sigma}$ 总是对称半正定，但未必可逆。它正定时，$\boldsymbol{e}^{\mathsf T}\boldsymbol{\Sigma}^{-1}\boldsymbol{e}$ 是平方马氏距离，用各方向的不确定性度量误差。这里先认识二次型，概率解释留到后面。

## 1.1.4 矩阵分解与稳定求解

### 特征值、奇异值与条件数

若 $\boldsymbol{A}\boldsymbol{q}=\lambda\boldsymbol{q}$ 且 $\boldsymbol{q}\ne0$，称 $\lambda$ 为特征值、$\boldsymbol{q}$ 为特征向量。实对称矩阵可以写成

$$
\boldsymbol{A}=\boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\mathsf T},\qquad
\boldsymbol{Q}^{\mathsf T}\boldsymbol{Q}=\boldsymbol{I},\qquad
\boldsymbol{\Lambda}=\operatorname{diag}(\lambda_1,\ldots,\lambda_n).
$$

在正交坐标 $\boldsymbol{z}=\boldsymbol{Q}^{\mathsf T}\boldsymbol{x}$ 中，二次型成为 $\sum_i\lambda_i z_i^2$。因此，对称矩阵正定当且仅当所有特征值为正。一般矩阵未必能在实数域对角化，不能不加条件地套用这个分解。

任意实矩阵 $\boldsymbol{A}\in\mathbb R^{m\times n}$ 都有奇异值分解（SVD）：

$$
\boldsymbol{A}=\boldsymbol{U}\boldsymbol{S}\boldsymbol{V}^{\mathsf T}.
$$

$\boldsymbol{U},\boldsymbol{V}$ 为正交方阵，$\boldsymbol{S}\in\mathbb R^{m\times n}$ 的矩形对角线上是非负奇异值。非零奇异值个数等于秩，最大奇异值等于谱范数。很小的奇异值意味着某个输入方向在输出中几乎不可见，反推输入时容易放大噪声。

对可逆方阵，二范数条件数为

$$
\kappa_2(\boldsymbol{A})=\|\boldsymbol{A}\|_2\|\boldsymbol{A}^{-1}\|_2
=\frac{\sigma_{\max}}{\sigma_{\min}}.
$$

当矩阵固定、$\boldsymbol{b}\ne0$ 且只有右端项受扰动时，$\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$ 的解满足

$$
\frac{\|\delta\boldsymbol{x}\|_2}{\|\boldsymbol{x}\|_2}
\leq\kappa_2(\boldsymbol{A})\frac{\|\delta\boldsymbol{b}\|_2}{\|\boldsymbol{b}\|_2}.
$$

这是最坏方向上的上界，不代表每个扰动都被同样放大。行列式受尺度影响很大，不宜仅凭“行列式很小”判断病态程度。

### 线性方程、最小二乘与伪逆

$\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$ 有解，当且仅当 $\boldsymbol{b}$ 属于列空间。方程比未知量多不代表一定无解，比未知量少也不代表一定有解；还要看秩和右端项。有解且零空间非平凡时，解有无穷多个。

没有精确解时，可以最小化 $\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2$。所有最小二乘解中，$\boldsymbol{A}^{\dagger}\boldsymbol{b}$ 是欧氏范数最小的一个，$\boldsymbol{A}^{\dagger}$ 为 Moore–Penrose 伪逆。SVD 对非零奇异值取倒数即可构造它；数值实现还需阈值判断哪些奇异值视为零。

| 问题 | 适合的计算方式 | 条件与用途 |
|---|---|---|
| 可逆方阵线性方程 | `solve(A, b)`，通常使用带选主元的分解 | 不必显式求逆 |
| 对称正定系统 | Cholesky：$\boldsymbol{A}=\boldsymbol{L}\boldsymbol{L}^{\mathsf T}$，再解两个三角系统 | 普通 Cholesky 要求正定 |
| 满列秩最小二乘 | 约化 QR：$\boldsymbol{A}=\boldsymbol{Q}\boldsymbol{R}$，解 $\boldsymbol{R}\boldsymbol{x}=\boldsymbol{Q}^{\mathsf T}\boldsymbol{b}$ | $\boldsymbol{Q}$ 的列正交归一 |
| 秩亏或接近秩亏的最小二乘 | 使用 SVD 的最小二乘或伪逆方法 | 明确奇异值截断阈值 |

正规方程 $\boldsymbol{A}^{\mathsf T}\boldsymbol{A}\boldsymbol{x}=\boldsymbol{A}^{\mathsf T}\boldsymbol{b}$ 适合推导，但满列秩时 $\kappa_2(\boldsymbol{A}^{\mathsf T}\boldsymbol{A})=\kappa_2(\boldsymbol{A})^2$。实际计算优先用分解或求解器，而不是照抄带逆矩阵的解析式。

这些分解足以支撑后续大部分估计与控制计算。Kronecker 积、一般张量代数和分解算法的底层实现暂不展开，需要时再引入。

## 1.1.5 求导前先确定输入、输出和形状

“对向量求导”可能指不同对象。设 $f:\mathbb R^n\to\mathbb R$，$\boldsymbol{g}:\mathbb R^n\to\mathbb R^m$：

| 对象 | 元素定义 | 形状 |
|---|---|---|
| 梯度 | $(\nabla_{\boldsymbol{x}}f)_i=\partial f/\partial x_i$ | $n\times1$ |
| 雅可比 | $(\boldsymbol{J}_{\boldsymbol{g}})_{ij}=\partial g_i/\partial x_j$ | $m\times n$ |
| 海森矩阵 | $(\boldsymbol{H}_f)_{ij}=\partial^2f/(\partial x_i\partial x_j)$ | $n\times n$ |
| 标量目标对矩阵的梯度 | $(\nabla_{\boldsymbol{X}}f)_{ij}=\partial f/\partial X_{ij}$ | 与 $\boldsymbol{X}$ 同形状 |

本手册的梯度是列向量，雅可比是**输出分量排在行，输入分量排在列**。所以标量函数的雅可比是行向量，$\boldsymbol{J}_f=(\nabla_{\boldsymbol{x}}f)^{\mathsf T}$，不是列梯度本身。

不同教材会采用不同布局，有些写 $\partial\boldsymbol{g}/\partial\boldsymbol{x}^{\mathsf T}$ 表示雅可比。不要仅凭分母有没有转置判断结果，先看元素定义。本手册优先使用 $\nabla f$ 和 $\boldsymbol{J}_{\boldsymbol{g}}$ 消除歧义。

“梯度与变量同形状”适用于这里的**标量目标对变量求导**。矩阵输出对矩阵输入的导数通常是线性算子；逐元素展开会有四个下标，不能仍当作一个同形状矩阵。

## 1.1.6 用微分法识别梯度

### 微分是局部线性变化

对可微标量函数有

$$
f(\boldsymbol{x}+\Delta\boldsymbol{x})=f(\boldsymbol{x})
+(\nabla_{\boldsymbol{x}}f)^{\mathsf T}\Delta\boldsymbol{x}
+o(\|\Delta\boldsymbol{x}\|_2).
$$

小 $o$ 表示余项除以 $\|\Delta\boldsymbol{x}\|_2$ 后趋于零。把线性部分写成微分：

$$
\boxed{\mathrm{d}f=(\nabla_{\boldsymbol{x}}f)^{\mathsf T}\mathrm{d}\boldsymbol{x}}.
$$

这个等号描述线性微分，不表示有限改变量总等于一阶项。沿方向 $\boldsymbol{v}$ 的方向导数为

$$
D_{\boldsymbol{v}}f=\left.\frac{\mathrm{d}}{\mathrm{d}\varepsilon}
f(\boldsymbol{x}+\varepsilon\boldsymbol{v})\right|_{\varepsilon=0}
=(\nabla f)^{\mathsf T}\boldsymbol{v}.
$$

定义不要求 $\boldsymbol{v}$ 长度为 1；只有解释为“单位距离上的变化率”时才需要单位化。由柯西–施瓦茨不等式，在欧氏单位方向中，非零梯度方向使一阶增长最快。对于混合单位的状态，先规定缩放或度量，再谈“最快方向”。

### 矩阵使用 Frobenius 内积

定义 $\langle\boldsymbol{A},\boldsymbol{B}\rangle_F=\operatorname{tr}(\boldsymbol{A}^{\mathsf T}\boldsymbol{B})=\sum_{i,j}A_{ij}B_{ij}$，则

$$
\boxed{\mathrm{d}f=\operatorname{tr}\bigl((\nabla_{\boldsymbol{X}}f)^{\mathsf T}\mathrm{d}\boldsymbol{X}\bigr)}.
$$

如果整理出的结果是 $\mathrm{d}f=\operatorname{tr}(\boldsymbol{B}\,\mathrm{d}\boldsymbol{X})$，梯度应为 $\boldsymbol{B}^{\mathsf T}$，而不是 $\boldsymbol{B}$。这是很多转置错误的来源。

### 可重复使用的五个步骤

1. 写出各量的形状，说明对谁求导、哪些量保持不变。
2. 对表达式取微分，对每个依赖求导变量的因子使用乘积法则。
3. 保持矩阵乘法顺序，必要时把标量写成迹。
4. 整理为 $\boldsymbol{g}^{\mathsf T}\mathrm{d}\boldsymbol{x}$ 或 $\operatorname{tr}(\boldsymbol{G}^{\mathsf T}\mathrm{d}\boldsymbol{X})$。
5. 读出梯度并检查形状，再用分量展开或数值方向导数核对。

迹允许循环移位：$\operatorname{tr}(\boldsymbol{A}\boldsymbol{B}\boldsymbol{C})=\operatorname{tr}(\boldsymbol{B}\boldsymbol{C}\boldsymbol{A})$，前提是乘积形状适配。它不允许任意交换因子。另一个常用事实是：标量等于自己的转置。

## 1.1.7 线性函数与二次型的推导

若 $\boldsymbol{a}$ 不依赖 $\boldsymbol{x}$，由 $\mathrm{d}(\boldsymbol{a}^{\mathsf T}\boldsymbol{x})=\boldsymbol{a}^{\mathsf T}\mathrm{d}\boldsymbol{x}$ 可直接读出梯度 $\boldsymbol{a}$。

现在令 $f=\boldsymbol{x}^{\mathsf T}\boldsymbol{A}\boldsymbol{x}$，固定 $\boldsymbol{A}$。左右两处 $\boldsymbol{x}$ 都会变化：

$$
\begin{aligned}
\mathrm{d}f
&=(\mathrm{d}\boldsymbol{x})^{\mathsf T}\boldsymbol{A}\boldsymbol{x}
+\boldsymbol{x}^{\mathsf T}\boldsymbol{A}\,\mathrm{d}\boldsymbol{x}\\
&=(\boldsymbol{A}\boldsymbol{x})^{\mathsf T}\mathrm{d}\boldsymbol{x}
+(\boldsymbol{A}^{\mathsf T}\boldsymbol{x})^{\mathsf T}\mathrm{d}\boldsymbol{x}.
\end{aligned}
$$

所以

$$
\boxed{\nabla_{\boldsymbol{x}}(\boldsymbol{x}^{\mathsf T}\boldsymbol{A}\boldsymbol{x})
=(\boldsymbol{A}+\boldsymbol{A}^{\mathsf T})\boldsymbol{x}}.
$$

只有 $\boldsymbol{A}$ 对称时，才可写成 $2\boldsymbol{A}\boldsymbol{x}$。二次型实际只取决于矩阵的对称部分：$\boldsymbol{x}^{\mathsf T}\boldsymbol{A}\boldsymbol{x}=\boldsymbol{x}^{\mathsf T}(\boldsymbol{A}+\boldsymbol{A}^{\mathsf T})\boldsymbol{x}/2$。

例如

$$
\boldsymbol{A}=\begin{bmatrix}2&3\\0&1\end{bmatrix},\qquad
f=2x_1^2+3x_1x_2+x_2^2.
$$

分量求导给出 $[4x_1+3x_2,\ 3x_1+2x_2]^{\mathsf T}$。在 $\boldsymbol{x}=[1,2]^{\mathsf T}$ 处，正确梯度为 $[10,7]^{\mathsf T}$；误用 $2\boldsymbol{A}\boldsymbol{x}$ 则得到 $[16,4]^{\mathsf T}$。

## 1.1.8 雅可比、链式法则与模型线性化

对 $\boldsymbol{y}=\boldsymbol{g}(\boldsymbol{x})$，微分关系为

$$
\mathrm{d}\boldsymbol{y}=\boldsymbol{J}_{\boldsymbol{g}}\,\mathrm{d}\boldsymbol{x}.
$$

例如 $\boldsymbol{y}=\boldsymbol{A}\boldsymbol{x}+\boldsymbol{b}$ 的雅可比为 $\boldsymbol{A}$。若再计算标量目标 $L(\boldsymbol{y})$，则

$$
\mathrm{d}L=(\nabla_{\boldsymbol{y}}L)^{\mathsf T}\boldsymbol{J}_{\boldsymbol{g}}\mathrm{d}\boldsymbol{x},
\qquad
\boxed{\nabla_{\boldsymbol{x}}L=\boldsymbol{J}_{\boldsymbol{g}}^{\mathsf T}\nabla_{\boldsymbol{y}}L}.
$$

转置来自列梯度约定：$(n\times m)(m\times1)=n\times1$。如果外层也是向量函数 $\boldsymbol{z}=\boldsymbol{h}(\boldsymbol{y})\in\mathbb R^p$，则

$$
\boldsymbol{J}_{\boldsymbol{h}\circ\boldsymbol{g}}(\boldsymbol{x})
=\underbrace{\boldsymbol{J}_{\boldsymbol{h}}(\boldsymbol{g}(\boldsymbol{x}))}_{p\times m}
\underbrace{\boldsymbol{J}_{\boldsymbol{g}}(\boldsymbol{x})}_{m\times n}.
$$

雅可比把输入扰动传到输出；它的转置把输出目标的敏感度传回输入。两种方向分别对应后面的 JVP 和 VJP。

如果非线性函数逐元素作用，即 $y_i=\phi(x_i)$，不同分量互不影响，因此 $\boldsymbol{J}=\operatorname{diag}(\phi'(x_1),\ldots,\phi'(x_n))$。若 $\boldsymbol{y}=\phi(\boldsymbol{A}\boldsymbol{x}+\boldsymbol{b})$，令 $\boldsymbol{z}=\boldsymbol{A}\boldsymbol{x}+\boldsymbol{b}$，则 $\boldsymbol{J}_{\boldsymbol{y}}=\operatorname{diag}(\phi'(z_1),\ldots,\phi'(z_m))\boldsymbol{A}$。这条规则连接了线性模型与神经网络中的非线性层。

### 时间变化与状态变化要一起计入

若 $L=L(t,\boldsymbol{x}(t))$，则

$$
\frac{\mathrm{d}L}{\mathrm{d}t}=\frac{\partial L}{\partial t}
+(\nabla_{\boldsymbol{x}}L)^{\mathsf T}\dot{\boldsymbol{x}}.
$$

偏导 $\partial L/\partial t$ 保持状态不变，全导数还计入状态沿轨迹变化的影响。

对动力学 $\dot{\boldsymbol{x}}=\boldsymbol{f}(\boldsymbol{x},\boldsymbol{u},t)$，沿满足动力学的参考轨迹定义扰动，得到一阶模型

$$
\delta\dot{\boldsymbol{x}}\approx\boldsymbol{A}(t)\delta\boldsymbol{x}
+\boldsymbol{B}(t)\delta\boldsymbol{u},
$$

$$
A_{ij}=\left.\frac{\partial f_i}{\partial x_j}\right|_{\rm ref},\qquad
B_{ij}=\left.\frac{\partial f_i}{\partial u_j}\right|_{\rm ref}.
$$

若参考状态与输入不满足原动力学，还需保留常数偏差项。雅可比得到的是局部模型，不保证大幅机动时仍准确；姿态也需要合适的局部坐标。

## 1.1.9 完整算例：测量误差与加权最小二乘

设测量 $\boldsymbol{y}\in\mathbb R^m$、固定测量矩阵 $\boldsymbol{H}\in\mathbb R^{m\times n}$、待估状态 $\boldsymbol{x}$。定义残差和目标：

$$
\boldsymbol{r}=\boldsymbol{H}\boldsymbol{x}-\boldsymbol{y},\qquad
L=\frac12\boldsymbol{r}^{\mathsf T}\boldsymbol{W}\boldsymbol{r},\qquad
\boldsymbol{W}=\boldsymbol{W}^{\mathsf T}\succ0.
$$

固定 $\boldsymbol{W}$，先对残差求导，再经过测量映射：

$$
\nabla_{\boldsymbol{r}}L=\boldsymbol{W}\boldsymbol{r},\qquad
\boldsymbol{J}_{\boldsymbol{r}}=\boldsymbol{H},\qquad
\boxed{\nabla_{\boldsymbol{x}}L=\boldsymbol{H}^{\mathsf T}\boldsymbol{W}(\boldsymbol{H}\boldsymbol{x}-\boldsymbol{y})}.
$$

$1/2$ 抵消了二次型求导产生的 2。若残差改成 $\boldsymbol{y}-\boldsymbol{H}\boldsymbol{x}$，其雅可比也变成 $-\boldsymbol{H}$，最终目标梯度不变。

取已经无量纲化的两维状态和三项测量：

$$
\boldsymbol{H}=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\quad
\boldsymbol{y}=\begin{bmatrix}1\\2\\2.8\end{bmatrix},\quad
\boldsymbol{W}=\operatorname{diag}(4,1,2).
$$

在 $\boldsymbol{x}_0=[0.8,1.9]^{\mathsf T}$ 处，残差为 $[-0.2,-0.1,-0.1]^{\mathsf T}$，目标 $L=0.095$，梯度为 $[-1,-0.3]^{\mathsf T}$。沿 $[1,0]^{\mathsf T}$ 方向的一阶变化率是 $-1$，所以小幅增大第一分量会降低目标。

令梯度为零：

$$
\begin{bmatrix}6&2\\2&3\end{bmatrix}\hat{\boldsymbol{x}}
=\begin{bmatrix}9.6\\7.6\end{bmatrix},\qquad
\hat{\boldsymbol{x}}=\begin{bmatrix}34/35\\66/35\end{bmatrix}
\approx\begin{bmatrix}0.971429\\1.885714\end{bmatrix}.
$$

左侧矩阵 $\boldsymbol{H}^{\mathsf T}\boldsymbol{W}\boldsymbol{H}$ 正定，所以解唯一。计算一般加权问题时，若 $\boldsymbol{W}=\boldsymbol{L}_W\boldsymbol{L}_W^{\mathsf T}$，可对 $\boldsymbol{L}_W^{\mathsf T}\boldsymbol{H}$ 与 $\boldsymbol{L}_W^{\mathsf T}\boldsymbol{y}$ 求普通最小二乘。若给的是协方差 $\boldsymbol{\Sigma}=\boldsymbol{L}_{\Sigma}\boldsymbol{L}_{\Sigma}^{\mathsf T}$，则用三角求解形成 $\boldsymbol{L}_{\Sigma}^{-1}\boldsymbol{H}$ 和 $\boldsymbol{L}_{\Sigma}^{-1}\boldsymbol{y}$，无需显式求逆。

## 1.1.10 矩阵变量求导：线性层与控制增益

### 参数矩阵的梯度为什么是外积

对 $\boldsymbol{y}=\boldsymbol{A}\boldsymbol{x}+\boldsymbol{b}$，令 $\boldsymbol{g}=\nabla_{\boldsymbol{y}}L$。允许三项都变化，有

$$
\mathrm{d}\boldsymbol{y}=(\mathrm{d}\boldsymbol{A})\boldsymbol{x}
+\boldsymbol{A}\,\mathrm{d}\boldsymbol{x}+\mathrm{d}\boldsymbol{b}.
$$

矩阵项可写成

$$
\boldsymbol{g}^{\mathsf T}(\mathrm{d}\boldsymbol{A})\boldsymbol{x}
=\operatorname{tr}(\boldsymbol{x}\boldsymbol{g}^{\mathsf T}\mathrm{d}\boldsymbol{A})
=\operatorname{tr}\bigl((\boldsymbol{g}\boldsymbol{x}^{\mathsf T})^{\mathsf T}\mathrm{d}\boldsymbol{A}\bigr).
$$

所以

$$
\nabla_{\boldsymbol{A}}L=\boldsymbol{g}\boldsymbol{x}^{\mathsf T},\qquad
\nabla_{\boldsymbol{x}}L=\boldsymbol{A}^{\mathsf T}\boldsymbol{g},\qquad
\nabla_{\boldsymbol{b}}L=\boldsymbol{g}.
$$

批量数据按样本求和；若目标是样本平均，还需除以样本数。

### 即时控制代价不等于整段轨迹代价

令 $\boldsymbol{u}=-\boldsymbol{K}\boldsymbol{e}$，其中 $\boldsymbol{K}\in\mathbb R^{m\times n}$，考虑

$$
L(\boldsymbol{K})=\frac12\boldsymbol{u}^{\mathsf T}\boldsymbol{R}\boldsymbol{u},\qquad
\boldsymbol{R}=\boldsymbol{R}^{\mathsf T}\succ0.
$$

**先固定状态误差 $\boldsymbol{e}$。** 因为 $\mathrm{d}\boldsymbol{u}=-(\mathrm{d}\boldsymbol{K})\boldsymbol{e}$，所以

$$
\nabla_{\boldsymbol{K}}L=-(\boldsymbol{R}\boldsymbol{u})\boldsymbol{e}^{\mathsf T}
=\boldsymbol{R}\boldsymbol{K}\boldsymbol{e}\boldsymbol{e}^{\mathsf T}.
$$

这只是固定状态下的即时敏感度。闭环轨迹的未来误差 $\boldsymbol{e}_k$ 也依赖 $\boldsymbol{K}$；整段飞行代价的梯度必须继续经过状态递推。漏掉这条路径，就是在求另一个问题。

### 矩阵到矩阵：不必展开四阶数组

若 $\boldsymbol{Y}=\boldsymbol{A}\boldsymbol{X}\boldsymbol{B}$，其中 $\boldsymbol{A}\in\mathbb R^{p\times m}$、$\boldsymbol{X}\in\mathbb R^{m\times n}$、$\boldsymbol{B}\in\mathbb R^{n\times q}$，则

$$
\mathrm{d}\boldsymbol{Y}=\boldsymbol{A}(\mathrm{d}\boldsymbol{X})\boldsymbol{B},\qquad
\frac{\partial Y_{ij}}{\partial X_{kl}}=A_{ik}B_{lj}.
$$

导数把 $m\times n$ 扰动映射成 $p\times q$ 输出扰动。若已知标量目标的输出梯度 $\boldsymbol{G}_Y$，利用迹循环移位可得

$$
\boxed{\nabla_{\boldsymbol{X}}L=\boldsymbol{A}^{\mathsf T}\boldsymbol{G}_Y\boldsymbol{B}^{\mathsf T}}.
$$

因此，计算标量目标梯度时通常无需显式存储四阶导数数组。

## 1.1.11 常用规则与适用条件

下表的常量不依赖求导变量，$\boldsymbol{X}^{-\mathsf T}$ 表示 $(\boldsymbol{X}^{-1})^{\mathsf T}$。

| 表达式 | 微分或梯度 | 条件 |
|---|---|---|
| $\boldsymbol{U}\boldsymbol{V}$ | $\mathrm{d}(\boldsymbol{U}\boldsymbol{V})=(\mathrm{d}\boldsymbol{U})\boldsymbol{V}+\boldsymbol{U}\,\mathrm{d}\boldsymbol{V}$ | 两因子可同时变化；顺序不变 |
| $\boldsymbol{X}^{\mathsf T}$ | $\mathrm{d}(\boldsymbol{X}^{\mathsf T})=(\mathrm{d}\boldsymbol{X})^{\mathsf T}$ | 任意实矩阵 |
| $\boldsymbol{X}^{-1}$ | $\mathrm{d}(\boldsymbol{X}^{-1})=-\boldsymbol{X}^{-1}(\mathrm{d}\boldsymbol{X})\boldsymbol{X}^{-1}$ | 可逆方阵 |
| $\operatorname{tr}(\boldsymbol{A}^{\mathsf T}\boldsymbol{X})$ | $\nabla_{\boldsymbol{X}}f=\boldsymbol{A}$ | $\boldsymbol{A}$ 为同形状常量 |
| $\boldsymbol{a}^{\mathsf T}\boldsymbol{X}\boldsymbol{b}$ | $\nabla_{\boldsymbol{X}}f=\boldsymbol{a}\boldsymbol{b}^{\mathsf T}$ | $\boldsymbol{a},\boldsymbol{b}$ 常量 |
| $\tfrac12\|\boldsymbol{X}\|_F^2$ | $\nabla_{\boldsymbol{X}}f=\boldsymbol{X}$ | 矩阵元素独立 |
| $\tfrac12\|\boldsymbol{A}\boldsymbol{X}-\boldsymbol{B}\|_F^2$ | $\nabla_{\boldsymbol{X}}f=\boldsymbol{A}^{\mathsf T}(\boldsymbol{A}\boldsymbol{X}-\boldsymbol{B})$ | $\boldsymbol{A},\boldsymbol{B}$ 常量 |
| $\log\det\boldsymbol{X}$ | $\nabla_{\boldsymbol{X}}f=\boldsymbol{X}^{-\mathsf T}$ | 实可逆方阵且 $\det\boldsymbol{X}>0$ |
| $\log\lvert\det\boldsymbol{X}\rvert$ | $\nabla_{\boldsymbol{X}}f=\boldsymbol{X}^{-\mathsf T}$ | 实可逆方阵，不可跨越奇异点 |

逆矩阵规则来自对 $\boldsymbol{X}\boldsymbol{X}^{-1}=\boldsymbol{I}$ 取微分：

$$
(\mathrm{d}\boldsymbol{X})\boldsymbol{X}^{-1}
+\boldsymbol{X}\,\mathrm{d}(\boldsymbol{X}^{-1})=\boldsymbol{0}.
$$

左乘 $\boldsymbol{X}^{-1}$ 即得结果。不能写成标量式的 $-\boldsymbol{X}^{-2}\mathrm{d}\boldsymbol{X}$，因为通常不能交换因子。

对数行列式的微分是 $\mathrm{d}\log\det\boldsymbol{X}=\operatorname{tr}(\boldsymbol{X}^{-1}\mathrm{d}\boldsymbol{X})$。后续协方差学习会使用这样的目标（固定 $\boldsymbol{e}$）：

$$
L(\boldsymbol{\Sigma})=\frac12\log\det\boldsymbol{\Sigma}
+\frac12\boldsymbol{e}^{\mathsf T}\boldsymbol{\Sigma}^{-1}\boldsymbol{e},\qquad\boldsymbol{\Sigma}\succ0.
$$

应用逆矩阵微分和迹的规则，得到

$$
\begin{aligned}
\mathrm{d}L
&=\tfrac12\operatorname{tr}(\boldsymbol{\Sigma}^{-1}\mathrm{d}\boldsymbol{\Sigma})
-\tfrac12\boldsymbol{e}^{\mathsf T}\boldsymbol{\Sigma}^{-1}(\mathrm{d}\boldsymbol{\Sigma})\boldsymbol{\Sigma}^{-1}\boldsymbol{e}\\
&=\tfrac12\operatorname{tr}\left[\left(\boldsymbol{\Sigma}^{-1}
-\boldsymbol{\Sigma}^{-1}\boldsymbol{e}\boldsymbol{e}^{\mathsf T}\boldsymbol{\Sigma}^{-1}\right)\mathrm{d}\boldsymbol{\Sigma}\right].
\end{aligned}
$$

括号内矩阵对称，转置后不变，因而可以读出

$$
\nabla_{\boldsymbol{\Sigma}}L=\frac12\left(\boldsymbol{\Sigma}^{-1}
-\boldsymbol{\Sigma}^{-1}\boldsymbol{e}\boldsymbol{e}^{\mathsf T}\boldsymbol{\Sigma}^{-1}\right).
$$

这是对称扰动下的 Frobenius 梯度。如果用独立参数存储对称矩阵，例如 $\boldsymbol{\Sigma}=\left[\begin{smallmatrix}a&b\\b&c\end{smallmatrix}\right]$，则 $\partial L/\partial b=G_{12}+G_{21}$，不能只取 $G_{12}$。对 Cholesky 因子等正定参数化求导，也必须继续使用链式法则。**完整矩阵梯度与独立参数梯度不是同一对象。**

## 1.1.12 海森矩阵与局部二次近似

若标量函数二阶连续可微，混合偏导可交换，海森矩阵对称，并且

$$
f(\boldsymbol{x}+\Delta\boldsymbol{x})=f(\boldsymbol{x})
+\boldsymbol{g}^{\mathsf T}\Delta\boldsymbol{x}
+\frac12\Delta\boldsymbol{x}^{\mathsf T}\boldsymbol{H}_f\Delta\boldsymbol{x}
+o(\|\Delta\boldsymbol{x}\|_2^2).
$$

梯度描述一阶变化，海森矩阵描述曲率。线性加权最小二乘有 $\boldsymbol{H}_L=\boldsymbol{H}^{\mathsf T}\boldsymbol{W}\boldsymbol{H}$。当 $\boldsymbol{W}\succ0$、$\boldsymbol{H}$ 满列秩时它正定；秩亏时只有半正定。

对非线性残差 $\boldsymbol{r}(\boldsymbol{x})$ 与固定对称权重，结果是

$$
\nabla L=\boldsymbol{J}_{\boldsymbol{r}}^{\mathsf T}\boldsymbol{W}\boldsymbol{r},
$$

$$
\boldsymbol{H}_L=\boldsymbol{J}_{\boldsymbol{r}}^{\mathsf T}\boldsymbol{W}\boldsymbol{J}_{\boldsymbol{r}}
+\sum_i(\boldsymbol{W}\boldsymbol{r})_i\nabla^2 r_i.
$$

只保留第一项是高斯–牛顿使用的曲率近似。线性残差时第二项为零；残差接近零且二阶导数有界时，第二项可能较小。一般情况下不能省略它后仍称其为精确海森矩阵。

在无约束光滑目标的内部局部极小点，梯度必须为零；若同时海森矩阵正定，则为严格局部极小点。仅有零梯度或半正定海森矩阵并不足够。具体算法留到 1.5，这里先理解算法使用的局部信息。

## 1.1.13 用 JAX 核对推导

以下代码可独立运行，需要 JAX 和 NumPy，不依赖 Aerodrome 实例。先预测结果，再运行核对。

[下载本页完整示例](/Aerodrome/examples/linear-algebra-checks.py)。

数学列向量在代码中用形状 `(n,)` 的一维数组表示。它没有独立行、列轴，`x.T` 仍是 `(n,)`；外积用 `jnp.outer`，显式列向量用 `x[:, None]`。`@` 表示矩阵乘法，`*` 表示逐元素乘法。

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

H = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
y = jnp.array([1., 2., 2.8])
W = jnp.diag(jnp.array([4., 1., 2.]))
x = jnp.array([0.8, 1.9])

def loss(x):
    r = H @ x - y
    return 0.5 * r @ W @ r

value, grad = jax.value_and_grad(loss)(x)
hessian = jax.hessian(loss)(x)
np.testing.assert_allclose(value, 0.095, atol=1e-12)
np.testing.assert_allclose(grad, [-1., -0.3], atol=1e-12)
np.testing.assert_allclose(hessian, H.T @ W @ H, atol=1e-12)

# 对角权重的白化最小二乘，不显式形成正规方程。
sqrt_w = jnp.sqrt(jnp.diag(W))
x_hat = jnp.linalg.lstsq(sqrt_w[:, None] * H, sqrt_w * y, rcond=None)[0]
np.testing.assert_allclose(x_hat, [34./35., 66./35.], atol=1e-12)

# 平面位置映射到距离和方位角；避开原点及角度分支切线。
def sensor(p):
    return jnp.array([jnp.linalg.norm(p), jnp.arctan2(p[1], p[0])])

p = jnp.array([3., 4.])
J = jax.jacfwd(sensor)(p)
np.testing.assert_allclose(J, [[0.6, 0.8], [-0.16, 0.12]], atol=1e-12)
v = jnp.array([0.2, -0.1])
_, jv = jax.jvp(sensor, (p,), (v,))
_, pullback = jax.vjp(sensor, p)
u = jnp.array([1., 0.5])
(jtu,) = pullback(u)
np.testing.assert_allclose(jv, J @ v, atol=1e-12)
np.testing.assert_allclose(jtu, J.T @ u, atol=1e-12)

# 中心差分只检查一个方向，无需构造全部差分梯度。
direction = jnp.array([0.6, -0.8])
eps = 1e-5
fd = (loss(x + eps * direction) - loss(x - eps * direction)) / (2 * eps)
np.testing.assert_allclose(fd, grad @ direction, rtol=1e-8, atol=1e-9)

# 固定状态下，对矩阵控制增益求导。
e = jnp.array([0.4, -0.2])
K = jnp.array([[2., -1.], [0.5, 3.]])
R = jnp.diag(jnp.array([2., 1.]))
def effort(K):
    control = -K @ e
    return 0.5 * control @ R @ control

np.testing.assert_allclose(jax.grad(effort)(K), R @ K @ jnp.outer(e, e), atol=1e-12)
print("通过：目标、梯度、海森矩阵、最小二乘、雅可比、JVP、VJP、方向差分、矩阵梯度")
```

`grad` 求标量目标梯度，`jacfwd`、`jacrev` 求雅可比；`jvp` 计算 $\boldsymbol{J}\boldsymbol{v}$，`vjp` 的回传函数计算 $\boldsymbol{J}^{\mathsf T}\boldsymbol{u}$。文献也常用行形式 $\boldsymbol{u}^{\mathsf T}\boldsymbol{J}$ 表示 VJP，两者互为转置。自动微分沿程序组合导数，不通过有限差分近似。[JAX 自动微分说明](https://docs.jax.dev/en/latest/jacobian-vector-products.html)

数值核对中启用 64 位可降低舍入误差，应在程序入口统一配置，不由库函数偷偷改变。[JAX 精度设置](https://docs.jax.dev/en/latest/101/default_dtypes.html)

只需一个方向的敏感度时用 JVP；已知输出目标的梯度、需回传到大量参数时用 VJP。不应为了一个乘积先构造巨大雅可比。模式选择取决于输入输出规模和所需导数，不能仅按有没有 GPU 决定。

中心差分的步长过大会产生截断误差，过小会放大浮点相消误差。应尝试相邻数量级并观察结果，尤其是变量尺度不同或接近不可微点时。自动微分吻合也不证明物理模型正确：单位、符号、坐标系仍需独立检查。饱和、碰撞、离散分支和外部 MATLAB 调用，都不能默认具有符合需求的可用梯度。

## 1.1.14 自检与练习

1. 两次位置测量满足 $\boldsymbol{y}=\left[\begin{smallmatrix}1&0\\1&\Delta t\end{smallmatrix}\right]\boldsymbol{x}_0$。什么时候能唯一确定初始位置和速度？为什么时间间隔很小时速度估计易受噪声影响？
2. 固定 $\boldsymbol{A},\boldsymbol{b}$，取 $\lambda>0$。求 $L=\tfrac12\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}\|_2^2+\tfrac\lambda2\|\boldsymbol{x}\|_2^2$ 的梯度和海森矩阵。为什么即使 $\boldsymbol{A}$ 秩亏，解仍唯一？
3. 给定 $\boldsymbol{y}=\boldsymbol{A}\boldsymbol{X}\boldsymbol{b}$、$L=\tfrac12\|\boldsymbol{y}-\boldsymbol{c}\|_2^2$，求 $\nabla_{\boldsymbol{X}}L$ 并标明形状。
4. 推导代码中距离、方位角传感器的雅可比，并说明哪些位置不满足光滑性前提。
5. 将控制增益扰动为 $\boldsymbol{K}\pm\varepsilon\boldsymbol{D}$，用中心差分检查是否等于 $\langle\nabla_{\boldsymbol{K}}L,\boldsymbol{D}\rangle_F$。

<details>
<summary>核对结果与提示</summary>

1. 当 $\Delta t\ne0$ 时矩阵满秩。速度等于位置差除以时间间隔，测量误差也被除以该间隔；数值条件还与状态缩放有关。
2. 梯度为 $\boldsymbol{A}^{\mathsf T}(\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b})+\lambda\boldsymbol{x}$；海森矩阵为 $\boldsymbol{A}^{\mathsf T}\boldsymbol{A}+\lambda\boldsymbol{I}\succ0$。
3. 令 $\boldsymbol{r}=\boldsymbol{A}\boldsymbol{X}\boldsymbol{b}-\boldsymbol{c}$，结果为 $\boldsymbol{A}^{\mathsf T}\boldsymbol{r}\boldsymbol{b}^{\mathsf T}$。若 $\boldsymbol{A}$ 是 $p\times m$、$\boldsymbol{X}$ 是 $m\times n$、$\boldsymbol{b}$ 是 $n\times1$，结果为 $m\times n$。
4. 令 $\rho=\sqrt{p_1^2+p_2^2}$，两行为 $[p_1/\rho,p_2/\rho]$ 和 $[-p_2/\rho^2,p_1/\rho^2]$。原点不可使用；`atan2` 的主值在负横轴有分支跳变，跨越它需角度展开或局部误差表示。
5. 比较差分与 `jnp.sum(jax.grad(effort)(K) * D)`。逐元素乘再求和才是 Frobenius 内积。

</details>

## 参考与后续阅读

- Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022，第 7 章，尤其 §7.7–7.8。本节核对的是所提供 PDF 中标注 **April 18, 2025** 的在线修订版，正文页码 229–274（PDF 文件页码 259–304）。本节的工程算例与逐步推导另行编写，未复用原书插图。
- JAX 官方文档：[雅可比乘积与自动微分](https://docs.jax.dev/en/latest/jacobian-vector-products.html)、[64 位精度开关](https://docs.jax.dev/en/latest/101/default_dtypes.html)。

在 [1.2 概率论](/Aerodrome/chapters/01-mathematical-tools/02-probability/) 中，我们会把状态和测量视为随机变量。本节的向量、二次型和线性映射，将用于表达均值、不确定性及其传播。
