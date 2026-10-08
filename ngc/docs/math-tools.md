# 数学工具算例：测量、拟合与多步选择

这两个程序对应[数学工具基础](/Aerodrome/chapters/01-mathematical-tools/)中的手算例子。默认数据是教学设定，模型和误差分布由我们指定，不是实际传感器记录。程序使用普通函数，测量与拟合例依赖 NumPy，动态规划例只需 Python 标准库；不需要机体、World 或训练框架。

## 运行方式

从仓库的 `ngc/` 目录执行，使用已安装 NumPy 的 Python 环境：

```bash
python examples/math_tools.py --seed 7 --repeats 2000
python examples/finite_horizon_dp.py
```

也可以在网站下载 [math_tools.py](/Aerodrome/downloads/math_tools.py) 与 [finite_horizon_dp.py](/Aerodrome/downloads/finite_horizon_dp.py)，在相同 Python 环境中直接运行下载文件。完整仓库已安装项目依赖时，无需另装库。

## 测量与拟合：同一组数在不同章节中的含义

`math_tools.py` 输出 JSON，字段分为四组。读输出前先预测数值变化，再用结果核对；浮点运算可能把 `0.8` 显示为 `0.7999999999999999`，这不改变下表中按给定精度报告的结果。

| 输出字段 | 默认结果（取适当小数） | 如何解释 |
|---|---|---|
| `measurement.weights` | `[0.8, 0.2]` | 标准差为 0.2 m 与 0.4 m 的独立、无偏测量，按逆方差加权 |
| `measurement.fused_position_m` | 9.88 m | 读数 9.8 m 与 10.2 m 的融合值 |
| `measurement.fused_variance_m2` | 0.032 m² | 由设定的独立误差模型算出的融合方差，不是这一次的实际误差平方 |
| `measurement.output_covariance_m2` | `[[7.4, 2.2], [2.2, 1]]` | 对二维误差作剪切变换后，计算得到的协方差 |
| `measurement.projection_m` | `[1.5, 1.5]` m | 将两次等精度读数 1、2 m 投影到共同位置模型；残差为 −0.5、0.5 m |
| `measurement.whitened_covariance` | 单位阵（允许浮点舍入误差） | 白化后的分量方差为 1、协方差为 0，不表示实际测量误差被消除 |
| `line_fit.ordinary` | 初始位置 1.016667 m，速度 2.05 m/s | 用 t=0、1、2 s 的读数 1.1、2.9、5.2 m 做最小二乘 |
| `line_fit.regularized` | 初始位置 1.041667 m，速度 2.025 m/s | 速度向参考 2 m/s 收缩；训练残差平方和从 0.041667 增加到 0.042917 m² |

`measurement_example` 对应[概率章的测量与协方差](/Aerodrome/chapters/01-mathematical-tools/02-probability/)；`fit_line` 对应[统计章的拟合与正则化](/Aerodrome/chapters/01-mathematical-tools/03-statistics/)。前者的两传感器误差设定，与后面的重复轨迹实验分开，修改 `--sigma` 只影响重复轨迹实验。

要改变两传感器各自的标准差，可以修改 `measurement_example` 中构造 `variances` 的 `[0.2, 0.4]`。输出的 `sample_output_covariance_m2` 来自另一个指定协方差的二维高斯抽样，与解析 `output_covariance_m2` 比较；`--seed` 和 `--repeats` 同时控制这组抽样与后面的重复拟合。两组计算各自创建随机数生成器，不把它们当作相互独立的联合实验。

## 重复实验：估计的不确定性和未来读数的波动

`repeated_predictions` 每轮生成一条新的三点轨迹。真实运动为 p(t)=1+2t，观测噪声独立且服从均值零、标准差 0.2 m 的高斯分布。程序每轮重新拟合，然后外推 t=3 s 的平均位置，并独立生成一次该时刻的新读数。

默认 2000 轮、seed=7 的本次运行结果如下。理论标准差来自指定模型，经验标准差来自程序本次生成的数据，两者不要求逐位相等。

| 比较量 | 经验标准差 | 理论标准差 |
|---|---:|---:|
| 拟合速度 | 0.141149 m/s | 0.141421 m/s |
| t=3 s 拟合均值相对真实均值的误差 | 0.306072 m | 0.305505 m |
| t=3 s 新读数相对拟合均值的误差 | 0.365897 m | 0.365148 m |

最后一行既包含拟合参数的不确定性，也包含新一次测量的独立噪声。单次读数相对真实均值的噪声标准差仍是 0.2 m。三者回答不同问题；这里不把某个标准差直接称为 95% 区间。

```bash
python examples/math_tools.py --seed 7 --repeats 2000 --sigma 0.4
python examples/math_tools.py --seed 7 --repeats 2000 --ridge-strength 0
```

第一条将噪声标准差加倍，理论上的三种标准差都应加倍；它没有改变采样时刻或轨迹模型。第二条取消正则化，`line_fit.regularized` 应与 `ordinary` 一致。增加 `--repeats` 是增加评估轮数，每轮仍只有三个观测点，因此只会让经验分布更稳定，不会降低单轮拟合的理论标准差。

## 决策：预测更准不总能换来更好的行动

`decision_examples` 对应[决策论](/Aerodrome/chapters/01-mathematical-tools/04-decision-theory/)。三个等可能处理时间为 2、2、8 s，均值输出 4 s，中位数输出 2 s；两种输出分别使平方损失和绝对损失最小。

分配例中，模型 A 的时间预测 RMSE 约为 0.424264 s，小于模型 B 的 2 s，但按预测选择任务后，A 的实际总处理时间是 7 s，B 是 6 s。B 把所有候选处理时间一起加了 2 s，不改变两种分配的排序；A 对交叉分配的误差较小，却足以改变排序。这个反例说明评价需要落到任务后果，不能推出预测误差不重要。

## 多步选择：先评价策略，再求更好的动作

`finite_horizon_dp.py` 对应[动态规划](/Aerodrome/chapters/01-mathematical-tools/06-dynamic-programming/)。状态是站点编号 0、1、2；动作是停留或前进一站，两步结束，每次前进花费 1，终端代价为 4(2−s)²。确定转移下，从终点倒推的最优代价应为：

```text
V2 = [16, 4, 0]
V1 = [5, 1, 0]
V0 = [2, 1, 0]
```

从状态 0 开始，一直停留的代价为 16，最优策略的代价为 2。第 0 步处于状态 1 时，停留和前进同样好；只保存一个动作不会使另一个变错。程序先评价给定策略，再进行倒推，可与正文表格逐项比较。

```bash
python examples/finite_horizon_dp.py --success-probability 0.8
```

随机转移下，尝试前进以 80% 概率成功、20% 概率留在原地，无论成功与否都花费 1。最后一步从状态 1 出发的最优期望代价为 1.8，从状态 0 出发为 7.4。这是在已知概率模型下计算期望，没有实际抽样运行很多次，也不保证某一次执行一定成功。

## 配图的来源

教材配图由 `website/scripts/plot-math-intuitions.py` 生成。重复拟合图直接调用本页的 `repeated_predictions`，其余几何图使用解析公式。需要重新生成时，从 `website/` 运行：

```bash
python scripts/plot-math-intuitions.py --font /path/to/chinese-font.ttf
```

绘图额外需要 Matplotlib；`--font` 指向本机支持中文的字体文件。普通读者运行上述两个算例只需 NumPy。返回[零号工程](/Aerodrome/reference/project-zero/#math-tools)。
