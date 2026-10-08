# 策略参数搜索与 REINFORCE

<span id="policy-learning"></span>

可单独下载 [policy_learning.py](/Aerodrome/downloads/policy_learning.py) 与 [dp_control.py](/Aerodrome/downloads/dp_control.py)，放在同一目录，使用已安装 NumPy 的 Python 运行。仓库内从 `ngc/` 运行：

```sh
uv run --locked python examples/policy_learning.py > policy-learning.json
```

脚本只依赖 NumPy，并从同目录的 `dp_control.py` 导入共同速度积分器 `trajectory`、有限时域 `finite_lqr` 与时间、限幅常量。输出是可直接解析的 JSON。`gain_search` 和 `comparison` 使用同一物理任务；`reinforce` 是单独的无量纲单步两动作问题，用来核对策略梯度符号，不能和速度控制代价横比。

## 速度控制和参数选择

状态 $x=v-v_{\rm ref}$（m/s），动作 $u$ 为加速度（m/s²），每步 $0.1$ s、共 $50$ 步，策略 $u=\operatorname{clip}(-Kx,-3,3)$。阶段无量纲代价为 $0.1[x^2+0.2u^2]$，终端代价 $x_{50}^2$；两个数值平方的参考单位分别是 $1$ m/s 和 $1$ m/s²，时间参考是 $1$ s。名义转移 $x_{k+1}=x_k+0.1u_k$。`gain_search()` 在训练初值 $1,2,3$ 上比较九个增益，在验证初值 $1.5,2.5,3.5$ 上从训练最佳三个候选中选择，再仅用 $-5,-4,4,5$ 报告留出代价。输出 `candidate_gain_per_s`、`train_cost`、`shortlist_gain_per_s`、`valid_cost`、`selected_gain_per_s`、`test_cost` 和 `selected_trace_x5`，轨迹内含每一步 `time_s`、`x_m_per_s`、`u_m_per_s2`。

实际运行选中 $K=2$ s⁻¹；四个测试代价依次为 $17.8940,10.0610,10.0610,17.8940$。固定 $K=0.5$ s⁻¹ 的对应代价为 $26.9117,17.2235,17.2235,26.9117$。`derivative_gain_per_s=0.8` 在训练初值上不会触及限幅；中心有限差分 `finite_difference=-3.13561167`，独立的无饱和解析式给出 `analytic_derivative=-3.13561161`，差约 $5.7\times10^{-8}$。这验证当前公式和积分器代码的衔接，不证明更复杂策略可微或能全局优化。

## 同一测试集上的三个控制器

`compare_controllers()` 复用 `dp_control.trajectory`，每个控制器都在 $-5,-4,4,5$ m/s 初值下运行。`fixed_gain_0_5` 不训练，`searched_gain` 在训练和验证初值上调参，`model_lqr_clipped` 使用名义模型与代价的有限时域 Riccati 递推，再在执行时限幅。三者的信息和离线计算量不同，不能把这张小表解释为算法类别的公平排名。

`comparison.controllers.nominal` 是名义模型；`mismatch_and_disturbance` 中实际转移改为 $x_{k+1}=x_k+0.1(0.8u_k)+0.02$，其中 $0.02$ 的单位是 m/s 每步。参数固定，不在变化后的模型上重训。名义工况三者平均代价依次约为 $22.0676,13.9775,13.9775$；变化工况依次约为 $28.0269,17.4059,17.4029$。每条轨迹另报告代价、前 50 个状态的 RMS 误差（m/s）、平均和最大控制幅值（m/s²）、终端绝对误差（m/s）、输入限幅违规步数。所有控制器均先限幅，当前违规步数为零。终端误差和有限时域代价是实测数值，不构成无限时域稳定性证明。

## 单步 REINFORCE 核对

另设 $a\in\{0,1\}$，代价 $c(0)=1,c(1)=0$，$p(a=1)=\sigma(\theta)$。精确目标 $J=1-p$，导数 $-p(1-p)$；采样梯度为 $(c(a)-b)(a-p)$，以 $\theta\leftarrow\theta-\eta\widehat{\nabla J}$ 更新。运行使用基线 $b=0.5$、学习率 $0.05$、种子 $7$、2000 次单步采样。初始 $\theta=0,p=0.5$；一次更新后 $\theta=0.0125,p\approx0.503125$。第 2000 次更新后实际输出 $p\approx0.989400$、精确期望代价 $\approx0.010600$。`reinforce.checkpoints` 同时提供 `exact_gradient`，便于核对代价最小化的方向；这些数字依赖随机种子和样本数，不能当作一般收敛率。

本例不实现 actor–critic、CEM 或 PPO。相关公式、假设与取舍见[3.6 直接优化控制策略](/Aerodrome/chapters/03-reinforcement-learning/06-policy-optimization/)；如何设计留出评测见[3.7 把控制方法放进同一实验](/Aerodrome/chapters/03-reinforcement-learning/07-control-experiments/)。
