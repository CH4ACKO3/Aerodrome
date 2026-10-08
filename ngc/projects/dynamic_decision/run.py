"""工程 02：外生参考跟踪与独立的三动作重复博弈。

两种任务只共用一次导出，不共享状态或奖励；函数按数据生成、训练、
闭环评估排列。直接运行本文件即可复现，算法只依赖 NumPy。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def reference(t, frequency, phase=0.0):
    """独立信号发生器；两种控制器均获得当前 r、r'、r''，没有未来预览。"""
    w = 2 * np.pi * frequency
    r = np.sin(w * t + phase) + 0.25 * np.sin(1.7 * w * t)
    velocity = w * np.cos(w * t + phase) + 0.25 * 1.7 * w * np.cos(1.7 * w * t)
    acceleration = -w**2 * np.sin(w * t + phase) - 0.25 * (1.7 * w)**2 * np.sin(1.7 * w * t)
    return r, velocity, acceleration


def collect_dynamics(seed, samples=2000):
    """独立台架样本覆盖速度和输入；监督标签是有小噪声的加速度测量。"""
    rng = np.random.default_rng(seed)
    velocity = rng.uniform(-2.5, 2.5, samples)
    command = rng.uniform(-6, 6, samples)
    features = np.column_stack((velocity, velocity**3, command))
    acceleration = features @ np.array([-0.4, -0.18, 1.0])
    return features, acceleration + rng.normal(0, 0.015, samples)


def tracking(frequency, phase, gains, model, drag_scale=1.0, duration=16.0, dt=0.02):
    """输入限幅下的二阶系统。真值积分使用 RK4，控制器每 dt 秒更新一次。

    model 保存 a = c1*v + c3*v^3 + b*u。PD 指定期望加速度，
    再用模型反解输入。传统模型忽略立方阻力，学习模型来自独立台架。
    """
    time_s = np.arange(round(duration / dt) + 1) * dt
    r, rv, ra = reference(time_s, frequency, phase)
    state = np.zeros((len(time_s), 2))
    command = np.zeros(len(time_s) - 1)
    kp, kd = gains
    for k in range(len(command)):
        position, velocity = state[k]
        desired = ra[k] + kp * (r[k] - position) + kd * (rv[k] - velocity)
        command[k] = np.clip((desired - model[0] * velocity - model[1] * velocity**3) / model[2], -6, 6)

        def derivative(x):
            v = x[1]
            return np.array([v, command[k] - drag_scale * (0.4 * v + 0.18 * v**3)])

        x = state[k]
        a = derivative(x)
        b = derivative(x + dt * a / 2)
        c = derivative(x + dt * b / 2)
        d = derivative(x + dt * c)
        state[k + 1] = x + dt * (a + 2*b + 2*c + d) / 6
    error = state[:, 0] - r
    metrics = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "input_rms": float(np.sqrt(np.mean(command**2))),
        "input_rate_rms": float(np.sqrt(np.mean((np.diff(command) / dt)**2))),
        "saturation_fraction": float(np.mean(np.abs(command) >= 6 - 1e-10)),
    }
    return time_s, state, command, r, metrics


def opponent_actions(seed, rounds, probability, switch=False):
    """外生 Markov 对手：倾向沿三动作循环，途中可改变方向。

    对手不读取学习者动作，因此所有方法能使用完全相同的动作序列。
    这是一类有限动作决策实验，不代表覆盖任意策略博弈。
    """
    rng = np.random.default_rng(seed)
    actions = np.zeros(rounds, dtype=int)
    actions[0] = rng.integers(3)
    for k in range(1, rounds):
        direction = -1 if switch and k >= rounds // 2 else 1
        actions[k] = (actions[k-1] + direction) % 3 if rng.random() < probability else rng.integers(3)
    return actions


def fit_transition(sequences, pseudocount):
    """监督学习：上一动作是特征，下一动作是标签；平滑频数即分类模型。"""
    counts = np.full((3, 3), pseudocount, dtype=float)
    for sequence in sequences:
        np.add.at(counts, (sequence[:-1], sequence[1:]), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def play(opponent, method, transition, seed):
    # 行是自己动作，列是对手动作；(+1) mod 3 的动作获得 +1。
    payoff = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    rng = np.random.default_rng(seed)
    actions = np.empty_like(opponent)
    for k in range(len(opponent)):
        if method == "uniform" or k == 0:
            actions[k] = rng.integers(3)
            continue
        if method == "empirical":
            # 传统滑动频率响应，只读取已经结束的轮次，不能偷看当前动作。
            probability = np.bincount(opponent[max(0, k-40):k], minlength=3) + 1
            probability = probability / probability.sum()
        else:
            probability = transition[opponent[k-1]]
        expected = payoff @ probability
        candidates = np.flatnonzero(np.isclose(expected, expected.max()))
        actions[k] = rng.choice(candidates)
    return actions, payoff[actions, opponent]


def run(output: Path, seed=7):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    features, labels = collect_dynamics(seed)
    validation_x, validation_y = collect_dynamics(seed + 1, 500)
    fit_log = []
    candidates = []
    for ridge in (0.0, 0.01, 1.0):
        coefficients = np.linalg.solve(features.T @ features + ridge * np.eye(3), features.T @ labels)
        mse = float(np.mean((validation_x @ coefficients - validation_y)**2))
        fit_log.append({"ridge": ridge, "validation_mse": mse})
        candidates.append(coefficients)
    model = candidates[int(np.argmin([row["validation_mse"] for row in fit_log]))]
    nominal = np.array([-0.4, 0.0, 1.0])

    # 控制增益只在验证参考调优，随后两种方法用完全相同的增益。
    gain_log = []
    for gains in ((2.0, 2.0), (4.0, 3.0), (6.0, 4.0)):
        scores = [tracking(f, 0.2, gains, nominal)[-1]["rmse"] for f in (0.07, 0.11)]
        gain_log.append({"gains": list(gains), "validation_rmse": float(np.mean(scores))})
    gains = min(gain_log, key=lambda row: row["validation_rmse"])["gains"]
    arrays = {"train_features": features, "train_acceleration": labels, "model_coefficients": model}
    evaluations = []
    for case, (frequency, drag_scale) in enumerate(((0.17, 1.0), (0.23, 1.0), (0.23, 1.2))):
        for method, coefficients in (("nominal_pd", nominal), ("identified_pd", model)):
            t, x, u, r, metrics = tracking(frequency, 0.4, gains, coefficients, drag_scale)
            key = f"tracking_{case}_{method}"
            arrays.update({key + "_state": x, key + "_command": u, key + "_reference": r})
            evaluations.append({"case": case, "frequency_hz": frequency, "drag_scale": drag_scale, "method": method, **metrics})
    arrays["tracking_time_s"] = t

    train_sequences = [opponent_actions(seed + i, 500, p) for i, p in enumerate((0.65, 0.75, 0.8))]
    validation = opponent_actions(seed + 100, 500, 0.7)
    game_tuning = []
    for smoothing in (0.1, 1.0, 10.0):
        transition = fit_transition(train_sequences, smoothing)
        nll = float(-np.mean(np.log(transition[validation[:-1], validation[1:]])))
        game_tuning.append({"pseudocount": smoothing, "validation_nll": nll})
    smoothing = min(game_tuning, key=lambda row: row["validation_nll"])["pseudocount"]
    transition = fit_transition(train_sequences, smoothing)
    arrays["learned_transition"] = transition
    arrays["game_training_actions"] = np.stack(train_sequences)
    game_rows = []
    for case, (probability, switch) in enumerate(((0.6, False), (0.9, False), (0.85, True))):
        for repeat in range(3):
            opponent = opponent_actions(seed + 200 + 10*case + repeat, 600, probability, switch)
            for method in ("uniform", "empirical", "learned_transition"):
                actions, reward = play(opponent, method, transition, seed + 500 + repeat)
                key = f"game_{case}_{repeat}_{method}"
                arrays.update({key + "_actions": actions, key + "_opponent": opponent, key + "_reward": reward})
                game_rows.append({"case": case, "repeat": repeat, "probability": probability, "switch": switch,
                                  "method": method, "mean_reward": float(reward.mean()),
                                  "first_half_reward": float(reward[:300].mean()),
                                  "second_half_reward": float(reward[300:].mean())})
    summary = {
        "project": "dynamic_decision", "version": 1, "seed": seed,
        "tracking": {"dt_s": 0.02, "duration_s": 16.0, "input_limit": 6.0,
                     "training_samples": len(labels), "training_velocity_range": [-2.5, 2.5],
                     "validation_frequencies_hz": [0.07, 0.11], "model_coefficients": model.tolist(),
                     "gain_tuning": gain_log, "fit_tuning": fit_log, "selected_gains": gains, "evaluation": evaluations},
        "game": {"training_rounds": 1500, "validation_rounds": 500, "evaluation_rounds": 600,
                 "tuning": game_tuning, "transition": transition.tolist(), "evaluation": game_rows},
        "elapsed_s": time.perf_counter() - started,
        "limitations": ["真实状态全观测，未加入噪声或缺测闭环", "博弈学习器冻结，方向切换会退化；未实现在线再训练", "没有飞行模型或空间交互"],
    }
    np.savez_compressed(output / "trajectory.npz", **arrays)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dynamic_decision"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run(args.output, args.seed)
    print(json.dumps({"output": str(args.output), "elapsed_s": result["elapsed_s"]}, ensure_ascii=False))
