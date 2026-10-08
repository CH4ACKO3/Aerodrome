"""可变成员集合：局部一致性与从示范拟合的共享线性策略。

直接运行本文件生成训练数据、留出评测和可供教程站播放的原始轨迹。
所有控制策略使用同一组特征，槽位序号只用于给定队形角色，不是观测。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from aerodrome.models.point_mass import PointMassState, step

DT = 0.05
DURATION = 18.0
CAPACITY = 9
RANGE_M = 4.5
LIMIT = 3.0
METHODS = ("independent", "consensus", "learned")


def reference(t):
    """所有节点接收相同的匀速参考；不需要知道全队人数。"""
    return np.array([0.35 * t, 0.0]), np.array([0.35, 0.0])


def observations(state, offsets, active, t):
    """固定容量只服务于记录，聚合始终排除自己和无效成员。

    每个节点只使用自身误差和通信范围内的邻居误差均值。均值而非
    求和使输入尺度不随人数线性增长，且与数组中的邻居排列无关。
    """
    target, target_v = reference(t)
    error = state.position_m - offsets - target
    velocity_error = state.velocity_m_s - target_v
    distances = np.linalg.norm(state.position_m[:, None] - state.position_m[None, :], axis=-1)
    neighbors = (distances < RANGE_M) & active[:, None] & active[None, :]
    np.fill_diagonal(neighbors, False)
    count = neighbors.sum(axis=1)
    # 没邻居时聚合项为零，仍由自身跟踪项推动运动；这是任务规则。
    mean_e = neighbors @ error / np.maximum(count[:, None], 1)
    mean_v = neighbors @ velocity_error / np.maximum(count[:, None], 1)
    features = np.stack((error, velocity_error,
                         np.where(count[:, None] > 0, error - mean_e, 0),
                         np.where(count[:, None] > 0, velocity_error - mean_v, 0)), axis=-1)
    return features, neighbors


def control(features, method, weights=None, coupling=1.0):
    """x/y 轴共享一组标量系数；先形成加速度，再使用共同输入上限。"""
    gains = {"independent": np.array([-1.4, -2.1, 0., 0.]),
             "consensus": np.array([-1.4, -2.1, -0.55 * coupling, -0.45 * coupling])}
    gains = weights if method == "learned" else gains[method]
    return np.clip(features @ gains, -LIMIT, LIMIT)


def simulate(count, seed, method, weights=None, events=True, permutation=None, coupling=1.0):
    """相同 seed 保证不同方法拥有相同初态、角色和成员事件。"""
    rng = np.random.default_rng(seed)
    offsets = np.column_stack((np.zeros(CAPACITY), (np.arange(CAPACITY) - 4) * 2.5))
    active = np.arange(CAPACITY) < count
    state = PointMassState(offsets + rng.normal(0, 0.35, (CAPACITY, 2)),
                           rng.normal(0, 0.08, (CAPACITY, 2)))
    ids = np.arange(CAPACITY)
    if permutation is not None:
        ids, offsets, active = ids[permutation], offsets[permutation], active[permutation]
        state = PointMassState(state.position_m[permutation], state.velocity_m_s[permutation])
    history = {key: [] for key in ("time", "position", "velocity", "action", "active", "features", "neighbors")}
    event_log = []
    interval_end_positions = []
    for k in range(round(DURATION / DT)):
        t = k * DT
        # 事件按稳定成员 ID 寻址，因此重新排列存储槽位不会改变实验。
        if events and k == round(6 / DT):
            active[ids == 1] = False
            event_log.append({"time_s": t, "kind": "leave", "member_id": 1})
        if events and k == round(10 / DT):
            joining = ids == count
            active[joining] = True
            p, v = reference(t)
            state.position_m[joining] = offsets[joining] + p + [0.6, 0.3]
            state.velocity_m_s[joining] = v
            event_log.append({"time_s": t, "kind": "join", "member_id": count})
        features, neighbors = observations(state, offsets, active, t)
        action = control(features, method, weights, coupling) * active[:, None]
        for key, value in zip(history, (t, state.position_m, state.velocity_m_s, action, active, features, neighbors)):
            history[key].append(np.array(value, copy=True))
        following = step(state, action, DT)
        # 退出成员不再参与物理、邻居或指标；保留槽位便于统一回放形状。
        state = PointMassState(np.where(active[:, None], following.position_m, state.position_m),
                               np.where(active[:, None], following.velocity_m_s, state.velocity_m_s))
        interval_end_positions.append(state.position_m.copy())
    trace = {key: np.asarray(value) for key, value in history.items()}
    trace["member_ids"] = ids
    trace["offsets"] = offsets
    trace["end_position"] = np.asarray(interval_end_positions)
    trace["end_time"] = trace["time"] + DT
    error = trace["features"][..., 0]
    squared_error = (error ** 2).sum(axis=-1)
    tail = trace["time"] >= 15
    pairs = trace["active"][:, :, None] & trace["active"][:, None, :] & ~np.eye(CAPACITY, dtype=bool)
    # 每个区间使用本步真实积分终点，不能直接连接相邻日志行：下一行
    # 可能已经执行加入/退出事件。这样退出前最后一个完整区间仍被评价，
    # 新成员在事件处的初始化跳变也不会变成一条虚假的飞行轨迹。
    relative = trace["position"][:, :, None] - trace["position"][:, None, :]
    relative_end = trace["end_position"][:, :, None] - trace["end_position"][:, None, :]
    delta = relative_end - relative
    fraction = np.clip(-np.sum(relative * delta, axis=-1) /
                       np.maximum(np.sum(delta ** 2, axis=-1), 1e-15), 0, 1)
    chord_minimum = np.linalg.norm(relative + fraction[..., None] * delta, axis=-1)
    relative_acceleration = trace["action"][:, :, None] - trace["action"][:, None, :]
    # ZOH相对运动是一条抛物线。它与端点弦的最大偏差为
    # ||a_i-a_j|| dt²/8，按三角不等式从弦最小距离扣除得到保守下界。
    deviation = np.linalg.norm(relative_acceleration, axis=-1) * DT ** 2 / 8
    lower_bound = np.maximum(chord_minimum - deviation, 0)
    trace["separation_lower_bound"] = np.min(np.where(pairs, lower_bound, np.inf), axis=(1, 2))
    min_distance = float(np.min(trace["separation_lower_bound"]))
    final_rmse = float(np.sqrt(np.mean(squared_error[tail][trace["active"][tail]])))
    rms_per_step = np.sqrt((squared_error * trace["active"]).sum(-1) / trace["active"].sum(-1))
    remains = np.logical_and.accumulate((rms_per_step < 0.12)[::-1])[::-1]
    recovery_indices = np.flatnonzero((trace["time"] >= 10) & remains)
    recovery = float(trace["time"][recovery_indices[0]] - 10) if len(recovery_indices) else None
    metrics = {"recovery_after_join_s": recovery, "count": count, "seed": seed, "method": method,
               "tracking_rmse_m": float(np.sqrt(np.mean(squared_error[trace["active"]]))),
               "final_rmse_m": final_rmse, "minimum_separation_m": min_distance,
               "separation_measure": "continuous ZOH conservative lower bound",
               "collision": bool(min_distance < 0.3), "completed": bool(final_rmse < 0.12 and min_distance >= 0.3),
               "mean_neighbors": float(np.mean(trace["neighbors"].sum(-1)[trace["active"]])),
               "mean_control_squared": float(np.mean((trace["action"] ** 2).sum(-1)[trace["active"]]))}
    return trace, metrics, event_log


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    # 训练只看2/4/6个成员；拟合目标是传统控制器的实际限幅后动作。
    xs, ys = [], []
    for count in (2, 4, 6):
        for seed in (11, 12, 13):
            trace, _, _ = simulate(count, seed, "consensus", events=False)
            xs.append(trace["features"][trace["active"]].reshape(-1, 4))
            ys.append(trace["action"][trace["active"]].reshape(-1))
    x, y = np.concatenate(xs), np.concatenate(ys)
    weights = np.linalg.lstsq(x, y, rcond=None)[0]
    np.savez_compressed(output / "training.npz", features=x, actions=y)
    np.savez(output / "policy.npz", weights=weights)
    records = []
    for count in (3, 5, 8):
        for seed in (101, 102):
            for method in METHODS:
                trace, metrics, events = simulate(count, seed, method, weights)
                name = f"{method}-n{count}-s{seed}"
                np.savez_compressed(output / f"{name}.npz", **trace)
                metrics["trace"] = name + ".npz"
                metrics["events"] = events
                records.append(metrics)
    # 不同存储顺序单独评估，不能用“人数不同”替代排列不变性验证。
    perm = np.random.default_rng(888).permutation(CAPACITY)
    ordinary, _, _ = simulate(5, 101, "learned", weights)
    reordered, _, _ = simulate(5, 101, "learned", weights, permutation=perm)
    permutation_error = float(np.max(np.abs(ordinary["position"] - reordered["position"][:, np.argsort(perm)])))
    tuning = []
    for coupling in (0.0, 0.5, 1.0, 1.5):
        _, metrics, _ = simulate(4, 31, "consensus", coupling=coupling)
        tuning.append({"coupling_scale": coupling, "tracking_rmse_m": metrics["tracking_rmse_m"],
                       "mean_control_squared": metrics["mean_control_squared"]})
    result = {"project": "variable_team", "tuning": tuning, "model": "planar double integrator, SI units",
              "training_rmse": float(np.sqrt(np.mean((x @ weights - y) ** 2))),
              "weights": weights.tolist(), "permutation_max_error_m": permutation_error,
              "evaluation": records, "elapsed_s": time.perf_counter() - started}
    config = {"dt_s": DT, "duration_s": DURATION, "capacity": CAPACITY, "neighbor_range_m": RANGE_M,
              "acceleration_limit_m_s2": LIMIT, "training_counts": [2, 4, 6], "training_seeds": [11, 12, 13],
              "test_counts": [3, 5, 8], "test_seeds": [101, 102], "version": 1}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    lines = ["# 可变规模协同：实际运行结果", "", "二维点质量模型；未声称四旋翼、障碍穿越或区域覆盖能力。", "",
             "| 方法 | 完成次数/6 | 平均末段误差(m) | 连续间距下界(m) |", "|---|---:|---:|---:|"]
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        lines.append(f"| {method} | {sum(r['completed'] for r in rows)}/6 | {np.mean([r['final_rmse_m'] for r in rows]):.5f} | {min(r['minimum_separation_m'] for r in rows):.3f} |")
    lines += ["", f"排列等价最大位置差：{permutation_error:.3g} m。", "共享线性策略拟合一致性示范，接近教师是预期结果，不代表超越教师。"]
    lines += ["", "## 单因素调优（4成员、seed=31，与留出集分开）", "", "| 邻居耦合倍率 | 全程RMSE(m) | 平均加速度平方 |", "|---:|---:|---:|"]
    lines += [f"| {r['coupling_scale']} | {r['tracking_rmse_m']:.5f} | {r['mean_control_squared']:.5f} |" for r in tuning]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/variable-team"))
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"output": str(args.output), "runs": len(result["evaluation"])}))
