# 统计学习算例：从模型表示到训练与评测

本页对应[第二章统计学习方法](/Aerodrome/chapters/02-statistical-learning/)。所有数据均为教学设定，入口为 `ngc/examples/statistical_learning.py`，只依赖 NumPy，不需要机体仿真器或深度学习框架。每个方法主题对应一个普通函数，便于直接看到输入、计算和输出；程序没有搭建通用训练框架。

## 运行方式

从仓库的 `ngc/` 目录执行，使用已安装 NumPy 的 Python：

```bash
python examples/statistical_learning.py --example logistic --seed 7
python examples/statistical_learning.py --example sequence
python examples/statistical_learning.py --example all > learning-results.json
```

也可下载 [statistical_learning.py](/Aerodrome/downloads/statistical_learning.py)，在同样的 Python 环境中直接运行。`--example` 选择下面登记的一个例子，默认 `all`；`--seed` 只影响逻辑回归例的合成数据和划分，其余例子是固定数值计算。输出 JSON 同时保留关键结果和绘图所需数组，完整输出较长，初读建议一次运行一个例子。

分类与回归模型只对下面明确说明的数据进行拟合。MLP、卷积、注意力和图聚合采用给定参数，展示一次运算或多次固定运算，不是已训练系统。矩阵补全只适用于本例刻意构造的低秩数据，不表示已有通用推荐服务。

<a id="lda"></a>

## 生成式分类：`--example lda`

`lda_demo()` 使用两个一维高斯分布，均值 0、2，方差均为 1，先验均为 0.5。输出类别条件密度和后验曲线；`probability_at_x_1_5` 约为 0.731059，等误分类代价边界为 1。没有从样本估计参数，属于解析分类例。

函数中的查询点后验由同一组密度计算，边界为等先验、共同方差下两均值的中点。修改先验或改用不同方差时，应相应改变边界计算。若要完整训练 LDA，应另用带标签的训练数据估计先验、均值与共同协方差。

<a id="logistic"></a>

## 逻辑回归：`--example logistic`

`logistic_demo(seed)` 生成 360 个独立二维高斯输入，并按给定逻辑概率随机生成 0/1 标签。随机置换后分为 216 个训练、72 个验证、72 个测试样本。标准化的均值与标准差只用训练集估计。每个候选模型由 `fit_logistic()` 做 1200 次全批量梯度下降，步长为 0.2；损失是平均负对数似然加 λ/2 倍斜率权重平方和，不惩罚截距。

三个正则化强度 0、0.03、0.3 在同一训练集上拟合，用验证集平均负对数似然选择一个。参数确定后，再读取测试标签并报告结果。本次实际运行 `--seed 7` 得到：

| 字段 | 结果（四舍五入） | 含义 |
|---|---:|---|
| `validation_nll` | [0.514755, 0.522342, 0.583453] | 三个候选模型在验证集的损失 |
| `selected_strength` | 0 | 这次验证划分选中的强度，不代表正则化普遍无用 |
| `test_nll` | 0.521897 | 测试集平均负对数似然，越低越好 |
| `test_accuracy` | 0.722222 | 阈值 0.5 下的分类准确率 |
| `test_brier` | 0.174847 | 概率与 0/1 标签的平均平方差 |
| `constant_baseline_test_nll` | 0.685599 | 恒定输出训练类别比例的基线 |

可以改种子重新生成数据，但不能挑出测试表现最好的种子作为最终结论。检查输出时先比较验证选择与测试评价的职责，再观察指标的有限样本波动。固定迭代次数是小例子的训练设置，不等于提供一般收敛保证。真实轨迹数据应按实验或时间划分，而非直接照搬这里独立样本的随机划分。

<a id="regression"></a>

## 线性回归：`--example regression`

`regression_demo()` 在五个无量纲样本上分别用 `[1,x]` 与 `[1,x,x²]` 拟合。直线系数为 `[2,-0.1]`，二次系数约为 `[0.057143,-0.1,0.971429]`，训练残差平方和分别为 13.24 与 0.028571。它展示特征决定可表达形状，没有独立测试集，因此不能据此宣称二次模型泛化更好。本例不实现 Lasso；岭回归的简单对照可回看[第一章数学工具算例](math-tools.md)。

<a id="glm"></a>

## 广义线性模型：`--example glm`

`glm_demo()` 采用一分钟窗口，无量纲负载 0、1、2 对应平均计数 2、4、8。零次概率依次约为 0.135335、0.018316、0.000335。这里只计算给定泊松模型，没有从计数记录训练系数。改变观察时长时要同时改变均值，而非把不同窗口的计数直接混在一起。

<a id="mlp"></a>

## 多层感知机：`--example mlp`

`mlp_demo()` 对四个 XOR 输入计算两项 ReLU 差分特征，给定输出权重后分类为 `[0,1,1,0]`。概率为约 `[0.119203,0.880797,0.880797,0.119203]`。可以打印 `hidden` 对照图中重合的类别 0 点；这个例子演示表示能力，不运行反向传播或训练。

<a id="convolution"></a>

## 卷积：`--example convolution`

`convolution_demo()` 将一个 3×2 差分模板作用于 5×5 图像，步幅 1、不补边，输出为 3×4，每行为 `[0,3,0,0]`。代码中 `patch * kernel` 对应局部逐元素乘积，再求和；模板没有翻转，因此是常见深度学习实现使用的互相关。所有像素值无量纲，未训练模板。

<a id="sequence"></a>

## 注意力：`--example sequence`

`sequence_demo()` 使用标量查询 1、键 `[1,0,2]`、值 `[10,20,30]`。先将未来位置得分置为负无穷，再逐行 softmax。输出约为 `[10,12.689414,24.205125]`。第三行权重约为 `[0.244728,0.090031,0.665241]`。这是一个因果注意力前向算例，不是完整 Transformer；观察移除掩码会不会让早期位置读取尚未发生的数据。

<a id="neighbors"></a>

## 近邻：`--example neighbors`

`neighbor_demo()` 使用四个一维训练点，在一组查询位置上分别计算 K=1、K=3 的预测。`three_neighbor_prediction_at_1_8` 为 1/3，单近邻在该处为 0。修改距离尺度、邻居数或训练点可观察阶梯怎样改变。多维输入需要先定义尺度；当前数组全部无量纲。

<a id="kernels"></a>

## 核与高斯过程：`--example kernels`

`kernel_demo(length=1)` 用四个观测计算零均值 GP 的后验，核幅度为 1，噪声方差为 0.04。`mean` 是潜在函数后验均值，`latent_sd` 是它的逐点标准差；新观测标准差要计算 `sqrt(latent_sd**2 + noise_variance)`。可在函数中改变长度尺度，比较数据间隔中的平滑程度与后验区间，不应把区间当成无条件有效的误差界。

<a id="trees"></a>

## 树与残差修正：`--example trees`

`best_stump()` 穷举一维训练输入相邻值的中点，选择训练残差平方和最小的切分。`tree_demo()` 首先得到阈值 2.5、训练 SSE 为 2/3 的树桩；再对残差拟合阈值 0.5 的树桩，以 0.5 学习率加入，SSE 变为 2/9。本程序不实现随机森林，第二棵树也不意味着在独立数据上一定更好。

<a id="augmentation"></a>

## 少标签与数据增强：`--example augmentation`

`augmentation_demo()` 对位置 `(1,0) m` 与速度标签 `(2,0) m/s` 使用同一个 90° 旋转矩阵，得到 `(0,1) m` 与 `(0,2) m/s`。速率标签仍为 2 m/s。它核对变换与标签的对应，不声称增强已改善一个学习任务，也不执行伪标签训练。

<a id="pca"></a>

## PCA：`--example pca`

`pca_demo()` 对四个二维样本做中心化与 SVD，一维主方向约为 `[0.906581,0.422033]`，保留方差比例约为 0.990326，平均重建平方误差约为 0.029651。特征向量正负号任意，代码只为绘图统一朝向。数据无量纲；该压缩指标不等于分类准确率。

<a id="clustering"></a>

## 聚类：`--example clustering`

`clustering_demo()` 对 `[0,1,4,5]` 使用初始中心 `[0,4]`，交替分配与更新三轮，得到 `[0.5,4.5]`，目标为 1。当前数据和初值不会产生空簇；这是固定算例，不是面向任意输入的聚类库。尝试别的初值时，应先手算分组并关注是否出现空簇，以及局部解是否变化。

<a id="recommendation"></a>

## 矩阵补全：`--example recommendation`

`recommendation_demo()` 构造秩一的 3×3 处理时间表，隐藏第二行第三列。交替最小二乘只读取 `observed_mask` 为真的八项，30 轮后预测约为 12 s，与保留真值相同。输出 `observed_values` 中的零是配合掩码存储的占位，不是观测值；内部训练数组在缺失处使用 NaN，训练只索引已观测项。因子没有正则化，且合成矩阵恰好满足秩一假设，不能将结果推广到真实稀疏推荐数据。

<a id="graphs"></a>

## 图聚合：`--example graphs`

`graph_demo()` 在三节点链上添加自连接，用各行度数归一化邻接矩阵。一轮特征从 `[0,2,4]` 变为 `[1,2,3]`，反复聚合后趋近 `[2,2,2]`。这里没有可学习权重，目的是观察邻域传播和差异消失。一般图的均值聚合极限受图结构影响，不保证总是原特征的简单算术平均。

## 配图与源码

15 幅图由 `website/scripts/plot-statistical-learning.py` 生成，直接调用上述各函数，并使用全站 `teaching.mplstyle`。从 `website/` 运行：

```bash
python scripts/plot-statistical-learning.py --font /path/to/chinese-font.ttf
npm run build
```

绘图额外需要 Matplotlib 和中文字体。图片查看器中的源码包包含绘图脚本、共用样式和算例程序，保留目录结构即可复现。修改算例数值以后，应同步核对章节手算结果、图注和本页的预期输出，再重新生成图片。

<a id="q-learning"></a>

## 强化学习：采样 Q 更新

本例是[强化学习简介](/Aerodrome/chapters/02-statistical-learning/05-reinforcement-learning/)之后可选的算法实验；详细原理将在第三章展开。从 `ngc/` 运行：

```bash
python examples/learning_control.py --example q-learning --seed 7
```

也可下载 [learning_control.py](/Aerodrome/downloads/learning_control.py)，只需 NumPy。`q_learning_demo()` 调用 `station_step()` 取得一次转移，按代价最小化的 Q 学习公式更新。站点 0、1 可等待或前进，站点 2 终止；每步代价为 1，折扣 0.9。模拟器可重置到任一非终止状态，因此均匀抽取状态—动作对，不是沿一条探索轨迹采样。

2000 次采样后，seed=7 的 `q` 非终止部分约为 `[[2.71,1.9],[1.9,1.0]]`，列依次为等待、前进。`policy=[1,1]`；`policy_values=[1.9,1.0]` 由独立的已知模型策略评价得到，而不是直接复制 Q 表。`visits` 显示各状态—动作的实际访问数；`samples` 和 `estimates` 保留绘图历史。`--seed` 只改变采样顺序。

初读可改变 `gamma`，将程序结果与两步到达的解析代价 `1+gamma` 比较。终止转移的目标只有本步代价；不能把“后续价值为零”改成额外的一步代价。该确定性小例没有展示随机转移、高维函数近似和现实探索成本。

<a id="parametric-control"></a>

## 参数化控制：按轨迹代价选增益

本例对应 [2.5 强化学习简介中的策略参数优化](/Aerodrome/chapters/02-statistical-learning/05-reinforcement-learning/#policy-optimization)。从 `ngc/` 运行：

```bash
python examples/learning_control.py --example control
python examples/learning_control.py --example all > learning-control-results.json
```

`velocity_rollout(K, initial_error)` 使用速度积分器、0.1 s 采样、5 s 实验长度和 ±3 m/s² 加速度限幅。它返回含终点的 `time`、`error`，每个区间实际执行的 `acceleration`，以及下面定义的无量纲 `objective`。控制量由本步开始误差计算，再更新下一步误差。

具体地，记速度误差为 $e_k$（m/s）、实际加速度为 $a_k=\operatorname{clip}(K e_k,-3,3)$（m/s²），更新为 $e_{k+1}=e_k-\Delta t\,a_k$。取 $\Delta t=0.1\,\mathrm s$、$N=50$，程序比较的轨迹代价为

$$
J(K)=\sum_{k=0}^{N-1}\frac{\Delta t}{t_0}
\left[\left(\frac{e_k}{e_{\mathrm{ref}}}\right)^2+
\rho\left(\frac{a_k}{a_{\mathrm{ref}}}\right)^2\right]
+\left(\frac{e_N}{e_{\mathrm{ref}}}\right)^2.
$$

这里 $e_{\mathrm{ref}}=1\,\mathrm{m/s}$、$a_{\mathrm{ref}}=1\,\mathrm{m/s^2}$、$t_0=1\,\mathrm s$ 是固定尺度，$\rho=0.2$，因此总代价无量纲。候选增益改变整条闭环轨迹；每条轨迹都用同一评价式计算，再按训练、验证和测试分工选择与评价参数。

`control_demo()` 在七个固定候选增益中搜索：训练初始误差为 `[1,3]` m/s，保留三个训练代价最小的候选 `[2,4,8]` s⁻¹；验证初始误差为 `[2,4]` m/s，平均代价约 `[6.045,6.150313,6.258438]`，选中 `selected_gain=2` s⁻¹。

固定增益后，用测试初值 `[-4,-1.5,5]` m/s 得到 `test_costs≈[10.061,1.125,17.894]`。预先固定的 K=0.5 基线得到 `baseline_costs≈[17.223482,2.422052,26.911691]`。两者使用同一时长、模型、目标尺度和输入限幅。实验没有随机扰动，`--seed` 不影响此例；不同初值也不代表不同动力学。

练习可改代价中的输入权重 0.2，重新选择参数并解释代价变化。比较不同权重的原始总分没有直接意义，因为评价目标已改变，应在同一个新目标下比较候选与基线。程序只实现比例控制和有限候选增益搜索，不包含 PID、LQR 或神经网络策略训练。

两幅图从同一程序输出生成。在 `website/` 中运行（需额外安装 Matplotlib）：

```bash
python scripts/plot-learning-control.py --font /path/to/chinese-font.ttf
```

<a id="mlp-backprop"></a>

## 神经网络：逐项核对前向、反向与参数更新

本例对应 [2.3 深度神经网络](/Aerodrome/chapters/02-statistical-learning/03-deep-neural-networks/)，与前面的固定权重 XOR 例子承担不同任务。它用两个输入、两个 ReLU 隐藏单元、一个线性输出，算出单样本平方损失及全部参数梯度，并更新一次参数。

从 `ngc/` 运行，只需 NumPy：

```bash
python examples/mlp_backprop.py
```

也可下载 [mlp_backprop.py](/Aerodrome/downloads/mlp_backprop.py) 后直接运行。输入 `[1,2]` 和目标 `1` 均已无量纲化；`W1` 为 2×2、`b1` 为长度 2、`W2` 为 1×2、`b2` 为长度 1。程序中的单样本向量是 NumPy 一维数组，教材数学上按列向量表示；`np.outer(delta, input)` 对应外积，`*` 对应逐元素相乘。

`forward()` 保存中间值；`backward()` 使用同一次前向的值与旧参数计算梯度；`demo()` 在全部梯度算好以后同时更新参数。`finite_difference()` 单独扰动每一个参数，只调用前向损失，作为反向传播的独立数值核对。固定例子的两个 ReLU 输入都严格大于零，差分没有跨过不可微点。

| 输出字段 | 默认结果 | 含义 |
|---|---|---|
| `before.z`、`before.hidden` | `[0.6,0.7]` | 隐藏层激活前后数值 |
| `before.prediction`、`before.loss` | `0.26`、`0.2738` | 更新前预测与半平方误差 |
| `gradients.W2`、`gradients.b2` | `[[-0.444,-0.518]]`、`[-0.74]` | 输出层参数梯度 |
| `gradients.W1`、`gradients.b1` | `[[-0.37,-0.74],[0.148,0.296]]`、`[-0.37,0.148]` | 隐藏层参数梯度 |
| `input_gradient` | `[-0.0814,-0.0148]` | 损失对输入的局部敏感度，输入不是本例待训练参数 |
| `max_gradient_error` | 约 `5.50e-11` | 步长 1e-6 中心差分与反传的最大绝对差，本机双精度运行结果 |
| `after.prediction`、`after.loss` | `0.53091696`、`0.1100194492` | 学习率 0.1、同时更新一次后的结果 |

损失在这个点的一次更新中下降，不表示任意学习率都下降，也不表示模型在新数据上有效。本例未做数据集训练、批量优化、Adam 或泛化评测。练习可修改学习率，比较更新前后损失；若改输入或参数使 ReLU 恰在零点，中心差分与程序选定的零点导数不必一致，应先确认求导条件。

计算链图直接读取该程序的实际结果。需要 NumPy、Matplotlib 和中文字体，在 `website/` 运行：

```bash
python scripts/plot-neural-computation.py --font /path/to/chinese-font.ttf
```

源码查看器的 `neural-computation.json` 包含绘图脚本、统一样式和该算例程序。
