"""有限时域速度控制与三站点采样值学习；只依赖 NumPy。

从 ngc/ 运行：uv run --locked python examples/dp_control.py > dp-control.json

速度误差 x=v-v_ref（m/s），控制 u 为加速度（m/s²）。每步零阶保持
dt=0.1 s，x_next=x+dt*u，|u|<=3。代价已按速度、加速度和时间标尺
无量纲化。无噪声、状态完全可见，限幅仅在执行和候选规划中使用。
三站点是另一个无量纲、折扣无限时域例子，不能和速度例比较代价值。
"""

import json
import numpy as np


DT = 0.1
T = 50
RHO = 0.2
LIMIT = 3.0


def stage_cost(x, u):
    """dt/t_ref * [(x/v_scale)^2 + rho (u/a_scale)^2]，三个标尺均为 1。"""
    return DT * (x * x + RHO * u * u)


def finite_lqr():
    """无输入限幅的有限时域 Riccati 倒推；P[T]=1 是终端成本。"""
    p = np.empty(T + 1)
    gain = np.empty(T)
    p[T] = 1.0
    for k in range(T - 1, -1, -1):
        # 对 dt(x²+rho u²)+P_next(x+dt u)² 求 u 的驻点。
        gain[k] = DT * p[k + 1] / (DT * RHO + DT * DT * p[k + 1])
        p[k] = DT + p[k + 1] - DT * p[k + 1] * gain[k]
    return p, gain


def base_action(x):
    """可行的基线反馈 u=clip(-x, -3, 3)，增益单位 s^-1。"""
    return float(np.clip(-x, -LIMIT, LIMIT))


def base_tail(k, x):
    """从第 k 步到 T 完整执行基线策略的剩余成本。"""
    cost = 0.0
    for _ in range(k, T):
        u = base_action(x)
        cost += stage_cost(x, u)
        x += DT * u
    return cost + x * x


def rollout_action(k, x):
    """一步候选后接完整基线；候选包含基线动作以保留改善条件。"""
    candidates = [-3.0, -1.5, 0.0, 1.5, 3.0, base_action(x)]
    return min(candidates, key=lambda u: stage_cost(x, u) + base_tail(k + 1, x + DT * u))


def mpc_action(k, x, p, horizon=3):
    """离散候选的短视域滚动规划，末端用无约束 LQR 值 P[k+h]x²。

    全部预测步强制 |u|<=3；只执行第一步。候选离散化使结果不是连续
    约束二次规划的精确解，也不据此声称 MPC 有稳定性保证。
    """
    h = min(horizon, T - k)
    candidates = [-3.0, -1.5, 0.0, 1.5, 3.0]

    def search(depth, state):
        if depth == h:
            return p[k + h] * state * state, None
        best = (float("inf"), None)
        for u in candidates:
            future, _ = search(depth + 1, state + DT * u)
            value = stage_cost(state, u) + future
            if value < best[0]:
                best = (value, u)
        return best

    return search(0, x)[1]


def trajectory(x0, policy, model_gain=1.0, disturbance=0.0):
    """运行 policy(x,k)，返回 x、u 与成本；扰动单位 m/s 每步。

    实际递推 x_next=x+dt*model_gain*u+disturbance。默认值是本章共同
    无扰动积分器。总成本始终用同一 stage_cost 和终端 x[T]² 计算。
    """
    states, actions = [float(x0)], []
    for k in range(T):
        u = float(policy(states[-1], k))
        actions.append(u)
        states.append(states[-1] + DT * model_gain * u + disturbance)
    cost = sum(stage_cost(x, u) for x, u in zip(states[:-1], actions)) + states[-1] ** 2
    return dict(x=states, u=actions, cost=cost)


def station_learning(seed=7, samples=2000):
    """无量纲三站点折扣无限时域；精确 Bellman 与采样 TD/Q 对照。

    站点 0、1 可等待或前进，到站点 2 终止；每步成本 1，gamma=0.9。
    TD(0) 评价“始终前进”，Q 学习从每个非终止状态动作对均匀采样。
    """
    rng = np.random.default_rng(seed)
    gamma = 0.9
    # 已知模型时，可对两状态 Bellman 算子反复作用来构造独立对照。
    exact_v = np.zeros(3)
    for _ in range(300):
        next_v = exact_v.copy()
        for s in (0, 1):
            next_v[s] = min(1 + gamma * exact_v[s],
                            1 if s == 1 else 1 + gamma * exact_v[s + 1])
        exact_v = next_v
    exact_q = np.array([[1 + gamma * exact_v[0], 1 + gamma * exact_v[1]],
                        [1 + gamma * exact_v[1], 1.0]])
    td = np.zeros(3)
    q = np.zeros((2, 2))
    td_visits = np.zeros(2, dtype=int)
    q_visits = np.zeros((2, 2), dtype=int)
    checkpoints = [0, 10, 50, 200, 1000, samples]
    history = []
    for n in range(samples + 1):
        if n in checkpoints:
            history.append(dict(samples=n, td=td.copy().tolist(), q=q.copy().tolist()))
        if n == samples:
            break
        # 评价目标策略“始终前进”：取到后继站点后仍按这一策略。
        s = int(rng.integers(0, 2))
        td_visits[s] += 1
        next_s = s + 1
        td_target = 1.0 if next_s == 2 else 1.0 + gamma * td[next_s]
        td[s] += td_visits[s] ** -0.6 * (td_target - td[s])
        # 行为分布均匀探索；Q 的目标选择后继的最低成本动作。
        s, a = (int(v) for v in rng.integers(0, 2, size=2))
        q_visits[s, a] += 1
        next_s = s + a
        q_target = 1.0 if next_s == 2 else 1.0 + gamma * q[next_s].min()
        q[s, a] += q_visits[s, a] ** -0.6 * (q_target - q[s, a])
    return dict(gamma=gamma, seed=seed, samples=samples,
                exact_v=exact_v.tolist(), exact_q=exact_q.tolist(),
                td=td.tolist(), q=q.tolist(), td_visits=td_visits.tolist(),
                q_visits=q_visits.tolist(), greedy_actions=q.argmin(axis=1).tolist(),
                history=history)


def main():
    p, gain = finite_lqr()
    controls = {
        "base": lambda x, k: base_action(x),
        "finite_lqr_unbounded": lambda x, k: -gain[k] * x,
        "finite_lqr_clipped": lambda x, k: np.clip(-gain[k] * x, -LIMIT, LIMIT),
        "rollout": lambda x, k: rollout_action(k, x),
        "mpc_grid_h3": lambda x, k: mpc_action(k, x, p),
    }
    initial_states = [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, -4.0, 4.0, 5.0]
    velocity = {str(x0): {name: trajectory(x0, control)
                          for name, control in controls.items()}
                for x0 in initial_states}
    output = dict(model=dict(dt_s=DT, steps=T, rho=RHO, limit_m_s2=LIMIT,
                             initial_states_m_s=initial_states),
                  riccati_p=p.tolist(), riccati_gain_per_s=gain.tolist(),
                  velocity=velocity, stations=station_learning())
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
