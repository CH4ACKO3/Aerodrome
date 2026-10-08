"""一维速度闭环的增益搜索与单步 REINFORCE；仅依赖 NumPy。

从 ngc/ 运行：python examples/policy_learning.py
速度误差 x=v-v_ref (m/s)，u 为加速度 (m/s²)。策略梯度另用
无量纲单步两动作问题，故两个目标值不能作算法性能横向比较。
"""

import json
import numpy as np
from dp_control import DT, T, LIMIT, trajectory, finite_lqr


def velocity_rollout(gain, initial_x, disturbance=0.0):
    """u=clip(-Kx, ±3 m/s²)；disturbance 为每步额外速度增量 (m/s)。

    阶段代价按 v_scale=1 m/s、a_scale=1 m/s²、t_ref=1 s
    无量纲化，终端再加 (x_T/v_scale)^2。
    """
    result = trajectory(initial_x,
                        lambda x, k: np.clip(-gain * x, -LIMIT, LIMIT),
                        disturbance=disturbance)
    return {"time_s": (DT * np.arange(T + 1)).tolist(),
            "x_m_per_s": result["x"], "u_m_per_s2": result["u"],
            "cost": result["cost"],
            "max_abs_u_m_per_s2": float(np.max(np.abs(result["u"])))}


def mean_cost(gain, initial_xs):
    return float(np.mean([velocity_rollout(gain, x)["cost"] for x in initial_xs]))


def gain_search():
    """训练格点提名三个增益，再按验证集选择；测试集只报告结果。"""
    train_x, valid_x, test_x = [1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [-5.0, -4.0, 4.0, 5.0]
    gains = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    train_costs = np.array([mean_cost(k, train_x) for k in gains])
    shortlist = gains[np.argsort(train_costs)[:3]]
    valid_costs = np.array([mean_cost(k, valid_x) for k in shortlist])
    selected = float(shortlist[np.argmin(valid_costs)])
    # 中心差分与无饱和闭环的解析导数相互核对；K=0.8 时训练初值均不饱和。
    derivative_gain = 0.8
    h = 1e-4
    fd = (mean_cost(derivative_gain + h, train_x)
          - mean_cost(derivative_gain - h, train_x)) / (2 * h)
    a = 1 - DT * derivative_gain
    n = np.arange(T)
    # 无饱和时 J(K,x0)=x0²[dt(1+0.2K²)sum_k(1-dt K)^{2k}
    #                         +(1-dt K)^{2T}]，直接微分该式。
    derivative_factor = DT * (0.4 * derivative_gain * np.sum(a ** (2 * n))
                              - DT * 2 * (1 + 0.2 * derivative_gain ** 2)
                              * np.sum(n[1:] * a ** (2 * n[1:] - 1)))
    derivative_factor -= DT * 2 * T * a ** (2 * T - 1)
    analytic = float(np.mean(np.square(train_x)) * derivative_factor)
    return {"train_initial_x_m_per_s": train_x, "valid_initial_x_m_per_s": valid_x,
            "test_initial_x_m_per_s": test_x, "candidate_gain_per_s": gains.tolist(),
            "train_cost": train_costs.tolist(), "shortlist_gain_per_s": shortlist.tolist(),
            "valid_cost": valid_costs.tolist(), "selected_gain_per_s": selected,
            "test_cost": [velocity_rollout(selected, x)["cost"] for x in test_x],
            "test_baseline_gain_per_s": 0.5,
            "test_baseline_cost": [velocity_rollout(0.5, x)["cost"] for x in test_x],
            "selected_trace_x5": velocity_rollout(selected, 5.0),
            "derivative_gain_per_s": derivative_gain,
            "finite_difference": fd, "analytic_derivative": analytic}


def compare_controllers(selected_gain=2.0):
    """相同测试初值下报告成本、跟踪 RMS、控制幅值与末态；仅限三种已实现控制器。"""
    _, lqr_gain = finite_lqr()
    controls = {
        "fixed_gain_0_5": lambda x, k: np.clip(-0.5 * x, -LIMIT, LIMIT),
        "searched_gain": lambda x, k: np.clip(-selected_gain * x, -LIMIT, LIMIT),
        "model_lqr_clipped": lambda x, k: np.clip(-lqr_gain[k] * x, -LIMIT, LIMIT),
    }
    cases = {"nominal": {"model_gain": 1.0, "disturbance": 0.0},
             "mismatch_and_disturbance": {"model_gain": 0.8, "disturbance": 0.02}}
    result = {}
    for case, settings in cases.items():
        result[case] = {}
        for name, control in controls.items():
            rows = []
            for x0 in (-5.0, -4.0, 4.0, 5.0):
                tr = trajectory(x0, control, **settings)
                x, u = np.asarray(tr["x"]), np.asarray(tr["u"])
                rows.append({"initial_x_m_per_s": x0, "cost": tr["cost"],
                             "rms_x_m_per_s": float(np.sqrt(np.mean(x[:-1] ** 2))),
                             "max_abs_x_m_per_s": float(np.max(np.abs(x))),
                             "mean_abs_u_m_per_s2": float(np.mean(np.abs(u))),
                             "max_abs_u_m_per_s2": float(np.max(np.abs(u))),
                             "terminal_abs_x_m_per_s": float(abs(x[-1])),
                             "limit_violation_steps": int(np.count_nonzero(np.abs(u) > LIMIT + 1e-12))})
            result[case][name] = {"per_initial_state": rows,
                                  "mean_cost": float(np.mean([r["cost"] for r in rows])),
                                  "mean_rms_x_m_per_s": float(np.mean([r["rms_x_m_per_s"] for r in rows]))}
    return {"cases": cases, "test_initial_x_m_per_s": [-5.0, -4.0, 4.0, 5.0],
            "controllers": result}


def reinforce_demo(seed=7, episodes=2000, learning_rate=0.05):
    """无量纲单步问题：动作 0 的成本 1，动作 1 的成本 0。

    πθ(1)=sigmoid(θ)，样本梯度=(C-b)(a-p)，按成本最小化更新。
    b=0.5 与当前动作无关；输出的解析梯度 p(1-p)(0-1) 可核对符号。
    """
    rng = np.random.default_rng(seed)
    theta = 0.0
    checkpoints = []
    for episode in range(episodes + 1):
        p = 1 / (1 + np.exp(-theta))
        if episode in (0, 1, 10, 100, 500, episodes):
            checkpoints.append({"episode": episode, "theta": theta,
                                "prob_action1": p, "exact_expected_cost": 1 - p,
                                "exact_gradient": -p * (1 - p)})
        if episode == episodes:
            break
        action = int(rng.random() < p)
        cost = 1 - action
        theta -= learning_rate * (cost - 0.5) * (action - p)
    return {"seed": seed, "episodes": episodes, "learning_rate": learning_rate,
            "baseline": 0.5, "checkpoints": checkpoints}


if __name__ == "__main__":
    gain_result = gain_search()
    print(json.dumps({"gain_search": gain_result,
                      "comparison": compare_controllers(gain_result["selected_gain_per_s"]),
                      "reinforce": reinforce_demo()},
                     ensure_ascii=False, indent=2))
