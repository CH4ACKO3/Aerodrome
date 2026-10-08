"""执行通道效能变化：固定反馈、理想补偿、在线辨识和数据估计。

对象为受恒定负载的一维双积分器。故障发生在指令到实际加速度之间；
除了显式命名的 oracle，方法不知道故障时间或真实效能。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from aerodrome.models.point_mass import PointMassState, step

DT = 0.05
DURATION = 22.0
LOAD = 1.2
NOISE = 0.002
LIMIT = 5.0
WINDOW = 24
METHODS = ("fixed", "oracle", "online", "learned")


def reference(t):
    """平滑外生参考及其解析导数，所有控制器得到相同参考。"""
    return (0.5 * np.sin(0.4 * t), 0.2 * np.cos(0.4 * t), -0.08 * np.sin(0.4 * t))


def make_training():
    """用独立效能水平和真实输入响应窗口训练一个小型近邻回归器。

    输入特征为平均指令、窗口平均响应；标签是数据采集时已知的效能。
    训练标签仅出现在离线采集，不会传入在线控制回路。
    """
    rng = np.random.default_rng(21)
    features, labels = [], []
    for efficacy in (0.45, 0.55, 0.75, 0.9, 1.0):
        for _ in range(240):
            command = rng.uniform(0.7, 4.5) + rng.normal(0, 0.08, WINDOW)
            state = PointMassState(np.array([0.0]), np.array([rng.normal()]))
            first = state.velocity_m_s[0] + rng.normal(0, NOISE)
            for u in command:
                state = step(state, np.array([efficacy * u - LOAD]), DT)
            last = state.velocity_m_s[0] + rng.normal(0, NOISE)
            response = (last - first) / (WINDOW * DT) + LOAD
            features.append([command.mean(), response])
            labels.append(efficacy)
    return np.asarray(features), np.asarray(labels)


def learned_estimate(feature, training, labels):
    """距离加权近邻回归，无故障标签或注入时间输入。

    两个特征均为m/s²量纲，训练范围相近，不额外引入归一化层。
    相邻训练效能间插值不是可靠外推：低于训练范围会留下误差。
    """
    distances = np.sum((training - feature) ** 2, axis=1)
    nearest = np.argpartition(distances, 11)[:12]
    weights = 1 / (distances[nearest] + 1e-4)
    return float(np.sum(weights * labels[nearest]) / weights.sum())


def simulate(method, efficacy_after, fault_time, seed, training, labels, kp=2.0):
    rng = np.random.default_rng(seed)
    state = PointMassState(np.array([0.0]), np.array([0.2]))
    estimate = 1.0
    commands, measurements = [], []
    rows = []
    for k in range(round(DURATION / DT)):
        t = k * DT
        efficacy = 1.0 if t < fault_time else efficacy_after
        measurement = state.velocity_m_s[0] + rng.normal(0, NOISE)
        measurements.append(measurement)
        # 窗口只含已经完成的动作/观测，估计先于当前新控制指令。
        # (v[k]-v[k-W])/(W*dt)+负载 = 窗口平均实际输入作用。
        if len(commands) >= WINDOW and method in ("online", "learned"):
            mean_u = float(np.mean(commands[-WINDOW:]))
            response = (measurements[-1] - measurements[-1 - WINDOW]) / (WINDOW * DT) + LOAD
            if method == "online":
                # 同一窗口内效能近似常数时，积分形式参数辨识避免速度差分放大噪声。
                estimate = float(np.clip(response / mean_u, 0.3, 1.1))
            else:
                estimate = learned_estimate(np.array([mean_u, response]), training, labels)
        if method == "oracle":
            estimate = efficacy
        r, rv, ra = reference(t)
        desired = ra + kp * (r - state.position_m[0]) + 2.5 * (rv - measurement)
        command = float(np.clip((desired + LOAD) / estimate, -LIMIT, LIMIT))
        commands.append(command)
        actual = efficacy * command
        rows.append([t, state.position_m[0], state.velocity_m_s[0], measurement, r, command,
                     actual, efficacy, estimate, desired])
        state = step(state, np.array([actual - LOAD]), DT)
    trajectory = np.asarray(rows)
    error = trajectory[:, 1] - trajectory[:, 4]
    after = trajectory[:, 0] >= fault_time
    late = trajectory[:, 0] >= fault_time + 5
    # 恢复定义为此后直到终点都保持误差带，不把瞬间过零称为恢复。
    within = np.abs(error) <= 0.06
    remains = np.logical_and.accumulate(within[::-1])[::-1]
    candidates = np.flatnonzero(after & remains)
    recovery = float(trajectory[candidates[0], 0] - fault_time) if len(candidates) else None
    metrics = {"method": method, "efficacy_after": efficacy_after, "fault_time_s": fault_time, "seed": seed,
               "before_rmse_m": float(np.sqrt(np.mean(error[~after] ** 2))),
               "after_rmse_m": float(np.sqrt(np.mean(error[after] ** 2))),
               "late_rmse_m": float(np.sqrt(np.mean(error[late] ** 2))),
               "peak_error_m": float(np.max(np.abs(error[after]))), "recovery_s": recovery,
               "late_efficacy_mae": float(np.mean(np.abs(trajectory[late, 8] - trajectory[late, 7]))),
               "saturation_fraction": float(np.mean(np.abs(trajectory[:, 5]) >= LIMIT - 1e-9)),
               "completed": bool(np.sqrt(np.mean(error[late] ** 2)) < 0.06)}
    return trajectory, metrics


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    training, labels = make_training()
    np.savez_compressed(output / "training.npz", features=training, efficacy=labels)
    # 近邻模型就是保存的训练特征/标签，不伪造一个未训练的网络权重文件。
    np.savez_compressed(output / "estimator.npz", features=training, efficacy=labels, neighbors=12)
    records = []
    scenarios = ((0.62, 7.0), (0.82, 10.0), (1.0, 8.0))
    for eta, fault_time in scenarios:
        for seed in (101, 102):
            for method in METHODS:
                trace, metrics = simulate(method, eta, fault_time, seed, training, labels)
                name = f"{method}-eta{eta:.2f}-s{seed}.npz"
                np.savez_compressed(output / name, trajectory=trace)
                metrics["trace"] = name
                records.append(metrics)
    # 调优只改变固定控制器的Kp，其余场景、噪声与输入上限均保持不变。
    # 此记录展示反馈增益增加与误差的联系，不用测试集挑选学习模型。
    tuning = []
    for kp in (1.0, 2.0, 3.0):
        _, metrics = simulate("fixed", 0.75, 6.0, 31, training, labels, kp=kp)
        tuning.append({"kp": kp, "late_rmse_m": metrics["late_rmse_m"], "saturation_fraction": metrics["saturation_fraction"]})
    config = {"version": 1, "dt_s": DT, "duration_s": DURATION, "load_m_s2": LOAD,
              "velocity_noise_std_m_s": NOISE, "input_limit_m_s2": LIMIT, "window_samples": WINDOW,
              "kp": 2.0, "kd": 2.5, "train_efficacies": [0.45, 0.55, 0.75, 0.9, 1.0], "train_seed": 21,
              "test_scenarios": [{"efficacy_after": e, "fault_time_s": t} for e, t in scenarios], "test_seeds": [101, 102],
              "trajectory_columns": ["time_s", "position_m", "velocity_m_s", "observed_velocity_m_s", "reference_m",
                                     "command_m_s2", "actual_input_m_s2", "true_efficacy", "estimated_efficacy", "desired_acceleration_m_s2"]}
    result = {"project": "adaptive_control", "evaluation": records, "tuning": tuning,
              "training_windows": len(training), "elapsed_s": time.perf_counter() - start}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    lines = ["# 故障与适应控制：实际运行结果", "", "一维点质量、恒定负载、输入效能下降。oracle拥有额外真值，仅是理想参照。", "",
             "| 方法 | 退化工况末段RMSE(m) | 正常工况末段RMSE(m) |", "|---|---:|---:|"]
    for method in METHODS:
        degraded = [r["late_rmse_m"] for r in records if r["method"] == method and r["efficacy_after"] < 1]
        normal = [r["late_rmse_m"] for r in records if r["method"] == method and r["efficacy_after"] == 1]
        lines.append(f"| {method} | {np.mean(degraded):.5f} | {np.mean(normal):.5f} |")
    lines += ["", "## 单因素调优记录", "", "| Kp | 固定反馈末段RMSE(m) | 饱和比例 |", "|---:|---:|---:|"]
    lines += [f"| {r['kp']} | {r['late_rmse_m']:.5f} | {r['saturation_fraction']:.3f} |" for r in tuning]
    lines += ["", "数据估计器在有限训练范围内插值；不保证比正确已知结构的在线辨识更好。传感器故障、卡滞、延迟和飞行器级故障尚未实现。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/adaptive-control"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"output": str(args.output), "runs": len(result["evaluation"])}))
