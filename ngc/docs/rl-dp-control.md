# 动态规划控制与采样值学习

<a id="dp-control"></a>

本例对应教材[第 3.4 节：rollout 与 MPC](/Aerodrome/chapters/03-reinforcement-learning/04-rollout-and-mpc/)和[第 3.5 节：值学习](/Aerodrome/chapters/03-reinforcement-learning/05-value-learning/)。源码是 [`examples/dp_control.py`](../examples/dp_control.py)。在 `ngc/` 目录运行：

```sh
uv run --locked python examples/dp_control.py > dp-control.json
```

也可下载 [dp_control.py](/Aerodrome/downloads/dp_control.py)，使用已安装 NumPy 的 Python 运行。脚本只依赖 NumPy，输出一个 JSON 对象，方便检查数值或绘制真实轨迹；它不训练神经网络，也不调用飞行器动力学。速度例和三站点例的状态、时域与代价定义不同，不直接比较二者的值。

## 速度积分器：输入、假设和公式

参考速度固定，状态为 $x_k=v_k-v_{\rm ref}$，单位 m/s；动作 $u_k$ 是加速度，单位 m/s²。$\Delta t=0.1$ s，$T=50$，默认实际模型与预测模型均为 $x_{k+1}=x_k+\Delta t u_k$，状态完全可见且没有过程噪声。受约束策略要求 $|u_k|\leq3$ m/s²。无量纲阶段代价是 $0.1(x_k^2+0.2u_k^2)$，终端代价为 $x_T^2$；这相当于速度、加速度、时间标尺分别取 $1$ m/s、$1$ m/s²、$1$ s。

`finite_lqr()` 从 $P_T=1$ 倒推无约束二次值 $V_k=P_kx^2$。把 $c(x,u)+P_{k+1}(x+\Delta t u)^2$ 对 $u$ 求导并令其为零，得到 $u=-K_kx$，其中

$$
K_k=\frac{\Delta t P_{k+1}}{\Delta t\,0.2+\Delta t^2P_{k+1}},\qquad
P_k=\Delta t+P_{k+1}-\Delta t P_{k+1}K_k.
$$

`base_action(x)` 是 $u=\operatorname{clip}(-x,-3,3)$。`base_tail(k,x)` 从该点完整执行基线到 $T$，包含最终 $x_T^2$。`rollout_action(k,x)` 比较有限候选动作的本步代价与其后基线代价，候选集特意保留当前的基线动作。`mpc_action(k,x,p,horizon=3)` 在每个预测步枚举 $\{-3,-1.5,0,1.5,3\}$ m/s²，只执行最优计划的第一步，末端值取无约束 $P_{k+h}x^2$。这个网格规划满足输入限幅，却不等于连续受约束优化的精确解。

`trajectory(x0, policy, model_gain=1.0, disturbance=0.0)` 是本章共同的实际执行入口。`policy(x,k)` 返回动作；返回对象含 `x`（51 个速度误差）、`u`（50 个加速度）、`cost`（同一代价公式）。它不自动裁剪动作，因为还要运行一个无约束 LQR 对照；需要满足执行限幅的策略须自行裁剪。可选的实际模型参数使递推变成 $x_{k+1}=x_k+\Delta t\,\text{model\_gain}\,u_k+\text{disturbance}$，其中 `disturbance` 单位 m/s 每步，默认均为 $1,0$。如果改变实际模型参数，`rollout_action` 和 `mpc_action` 的预测仍使用名义模型，因此输出代价是实际轨迹的实现代价，不能当作规划器的预测代价。

`velocity` 以初值字符串为键，包含 `base`、`finite_lqr_unbounded`、`finite_lqr_clipped`、`rollout`、`mpc_grid_h3` 五条策略轨迹。`model` 保存常数和初值；`riccati_p` 有 $51$ 项，`riccati_gain_per_s` 有 $50$ 项。初值包含训练侧 $1,2,3$、验证侧 $1.5,2.5,3.5$ 以及测试侧 $-4,4,5$ m/s。这里未通过这些集合训练或挑选参数；列出它们是为了与本章其他例子保持工况一致。

运行结果中，$x_0=1$ 时基线代价约 $0.631589$，rollout 约 $0.549725$；$x_0=3$ 时分别约 $5.684299$ 与 $5.029351$。无约束 LQR 在 $x_0=3$ 的首步动作约为 $-6$ m/s²，超过 $\pm3$，因此其代价 $4.5$ 不是可行受限控制器的成绩。限幅后的 LQR 代价约为 $4.995$，网格 MPC 约为 $5.1345$。这些数值是在同一名义模型上测得的一组工况，不提供模型失配下的保证。

## 三站点：精确值与采样值

`station_learning()` 是独立的无量纲折扣无限时域问题。站点 $0,1$ 可以等待（动作 $0$）或前进（动作 $1$）；到站点 $2$ 终止，每步代价 $1$，折扣 $\gamma=0.9$。精确 Bellman 迭代给出 $V^*=[1.9,1,0]$，$Q^*=[[2.71,1.9],[1.9,1]]$。它与 `learning_control.py` 的三站点 Q 学习采用同一任务，但另外加入 TD(0) 对照；它和 `finite_horizon_dp.py` 中“两次动作、到期终端罚分”的站点问题不同。

学习器每次均匀重置到站点 $0$ 或 $1$，TD(0) 评价“总是前进”的策略，Q 学习均匀抽取非终止状态动作对并使用最低代价自举目标。两者都只在更新器中使用采样的转移；精确 Bellman 表另作对照。访问某项第 $n$ 次时步长为 $n^{-0.6}$。默认随机种子 $7$，每种更新各取 $2000$ 个样本。`stations` 的 `td`、`q` 是最终估计，`td_visits`、`q_visits` 是计数，`history` 在样本数 $0,10,50,200,1000,2000$ 保存估计表；`exact_v`、`exact_q` 和 `greedy_actions` 便于核对。默认输出的贪心动作是 $[1,1]$，表示两处都前进。

可把 `station_learning(samples=50)` 与 `station_learning(samples=2000)` 的 `q` 同 `exact_q` 比较，检查误差是否集中在访问次数少的表项。环境确定、只有四个有效动作且允许重置，故本例很容易接近精确表；这不代表一般函数近似 Q 学习会同样收敛。
