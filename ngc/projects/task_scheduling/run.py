"""工程 04：不可拆分作业的匹配与在线调度。

每个处理节点容量为一；输入只有工作量、节点参数和时间。先学习处理耗时，
再比较贪心、标称线性分配与学习耗时驱动的线性分配，不建立调度框架层。
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment


def duration_features(work, speed, setup):
    """数组可广播；所有方法均能看到 work、speed、setup，无节点编号记忆。"""
    work, speed, setup = np.broadcast_arrays(work, speed, setup)
    return np.stack((work / speed, work**2 / speed, setup, np.ones_like(work)), axis=-1)


def actual_duration(work, speed, setup):
    # 仿真对象的真实处理耗时；调度器不能直接调用它作决策。
    return work / speed * (1 + 0.15 * work) + setup


def fit_predictor(seed):
    """真实监督回归：训练 1200 个独立已完成作业，验证集选择岭参数。"""
    rng = np.random.default_rng(seed)

    def sample(count, maximum):
        work = rng.uniform(1, maximum, count)
        speed = rng.uniform(0.8, 2.0, count)
        setup = rng.uniform(0.1, 1.0, count)
        x = duration_features(work, speed, setup)
        y = actual_duration(work, speed, setup) + rng.normal(0, 0.025, count)
        return x, y

    x, y = sample(1200, 6)
    vx, vy = sample(400, 8)
    tuning, candidates = [], []
    for ridge in (0.0, 0.01, 1.0):
        model = np.linalg.solve(x.T @ x + ridge * np.eye(4), x.T @ y)
        candidates.append(model)
        tuning.append({"ridge": ridge, "validation_mse": float(np.mean((vx @ model - vy)**2))})
    model = candidates[int(np.argmin([row["validation_mse"] for row in tuning]))]
    return model, tuning, x, y


def estimate(work, speed, setup, method, model):
    if method == "learned_assignment":
        return duration_features(work, speed, setup) @ model
    return work / speed + setup


def choose(cost, method):
    """所有方案都输出唯一节点—作业对；约束满足来自组合步骤，而非回归。"""
    if method != "greedy":
        return linear_sum_assignment(cost)
    remaining = cost.copy()
    rows, columns = [], []
    for _ in range(min(cost.shape)):
        row, column = np.unravel_index(np.argmin(remaining), remaining.shape)
        rows.append(row)
        columns.append(column)
        remaining[row, :] = np.inf
        remaining[:, column] = np.inf
    return np.asarray(rows), np.asarray(columns)


def scenario(seed, node_count, job_count, streaming):
    rng = np.random.default_rng(seed)
    speed = rng.uniform(0.8, 2.0, node_count)
    setup = rng.uniform(0.1, 1.0, node_count)
    work = rng.uniform(3, 10, job_count)
    arrival = np.sort(rng.uniform(0, 30, job_count)) if streaming else np.zeros(job_count)
    deadline = arrival + rng.uniform(4, 14, job_count)
    return speed, setup, work, arrival, deadline


def online_schedule(scene, method, model, lateness_weight=2.0):
    """事件驱动、无未来预知、开始后不抢占。

    一个事件是作业到达或节点完成。空闲节点仅能从已到达队列中选择；
    分配后以真实处理耗时推进时钟，预测误差不会被画图掩盖。
    """
    speed, setup, work, arrival, deadline = scene
    free_at = np.zeros(len(speed))
    start = np.full(len(work), np.nan)
    finish = np.full(len(work), np.nan)
    assigned_node = np.full(len(work), -1, dtype=int)
    now = 0.0
    while np.any(assigned_node < 0):
        waiting = np.flatnonzero((arrival <= now + 1e-10) & (assigned_node < 0))
        free = np.flatnonzero(free_at <= now + 1e-10)
        if len(waiting) and len(free):
            predicted = estimate(work[waiting][None, :], speed[free, None], setup[free, None], method, model)
            # 年龄奖励减少旧作业一直排队；截止惩罚是简单启发式，不宣称全局最优。
            cost = predicted + lateness_weight * np.maximum(now + predicted - deadline[waiting], 0)
            cost -= 0.5 * (now - arrival[waiting])
            local_nodes, local_jobs = choose(cost, method)
            nodes, jobs = free[local_nodes], waiting[local_jobs]
            assigned_node[jobs] = nodes
            start[jobs] = now
            finish[jobs] = now + actual_duration(work[jobs], speed[nodes], setup[nodes])
            free_at[nodes] = finish[jobs]
        # 只选严格未来事件；作业已经到达但尚未服务时等待最近节点完成。
        events = np.concatenate((arrival[(assigned_node < 0) & (arrival > now + 1e-10)], free_at[free_at > now + 1e-10]))
        if np.any(assigned_node < 0):
            now = float(events.min())
    waiting_time = start - arrival
    tardiness = np.maximum(finish - deadline, 0)
    metrics = {"completed_fraction": float(np.mean(np.isfinite(finish))),
               "mean_wait": float(waiting_time.mean()), "max_wait": float(waiting_time.max()),
               "mean_flow_time": float(np.mean(finish - arrival)), "mean_tardiness": float(tardiness.mean()),
               "deadline_violation_fraction": float(np.mean(tardiness > 0)), "makespan": float(finish.max()),
               "utilization": float(np.sum(finish - start) / (len(speed) * finish.max()))}
    return assigned_node, start, finish, metrics


def run(output: Path, seed=11):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    model, tuning, features, labels = fit_predictor(seed)
    arrays = {"model_coefficients": model, "train_features": features, "train_duration": labels}
    methods = ("greedy", "nominal_assignment", "learned_assignment")

    # 调参只用独立验证流，评测流和规模在下面另行生成。
    scheduling_tuning = []
    for weight in (0.0, 1.0, 2.0, 4.0):
        scores = []
        for repeat in range(3):
            scene = scenario(seed + 50 + repeat, 3, 18, True)
            metrics = online_schedule(scene, "nominal_assignment", model, weight)[-1]
            scores.append(metrics["mean_flow_time"] + metrics["mean_tardiness"])
        scheduling_tuning.append({"lateness_weight": weight, "validation_cost": float(np.mean(scores))})
    weight = min(scheduling_tuning, key=lambda row: row["validation_cost"])["lateness_weight"]

    static_rows = []
    for case, size in enumerate((3, 5, 8, 12)):
        scene = scenario(seed + 100 + case, size, size, False)
        speed, setup, work, _, _ = scene
        truth = actual_duration(work[None, :], speed[:, None], setup[:, None])
        oracle_nodes, oracle_jobs = linear_sum_assignment(truth)
        oracle_cost = float(truth[oracle_nodes, oracle_jobs].sum())
        # 小规模枚举是独立的组合数值校核，只在 3x3 和 5x5 执行。
        enumerated = min(float(truth[np.arange(size), p].sum()) for p in itertools.permutations(range(size))) if size <= 5 else None
        arrays[f"static_{case}_true_duration"] = truth
        arrays[f"static_{case}_work"] = work
        arrays[f"static_{case}_speed"] = speed
        arrays[f"static_{case}_setup"] = setup
        for method in methods:
            predicted = estimate(work[None, :], speed[:, None], setup[:, None], method, model)
            nodes, jobs = choose(predicted, method)
            arrays[f"static_{case}_{method}_pairs"] = np.column_stack((nodes, jobs))
            static_rows.append({"size": size, "method": method, "true_total_duration": float(truth[nodes, jobs].sum()),
                                "oracle_cost": oracle_cost, "enumerated_cost": enumerated,
                                "prediction_rmse": float(np.sqrt(np.mean((predicted - truth)**2)))})
    online_rows = []
    for case, (nodes, jobs) in enumerate(((3, 5), (3, 35), (5, 45), (8, 60))):
        for repeat in range(3):
            scene = scenario(seed + 200 + 10*case + repeat, nodes, jobs, True)
            prefix = f"online_{case}_{repeat}"
            for name, values in zip(("speed", "setup", "work", "arrival", "deadline"), scene):
                arrays[prefix + "_" + name] = values
            for method in methods:
                assigned, start, finish, metrics = online_schedule(scene, method, model, weight)
                key = prefix + "_" + method
                arrays.update({key + "_node": assigned, key + "_start": start, key + "_finish": finish})
                online_rows.append({"case": case, "repeat": repeat, "nodes": nodes, "jobs": jobs, "method": method, **metrics})
    summary = {"project": "task_scheduling", "version": 1, "seed": seed, "node_capacity": 1,
               "training_samples": len(labels), "training_work_range": [1, 6], "validation_work_range": [1, 8],
               "evaluation_work_range": [3, 10], "model_coefficients": model.tolist(), "fit_tuning": tuning,
               "scheduling_tuning": scheduling_tuning, "selected_lateness_weight": weight,
               "static_evaluation": static_rows, "online_evaluation": online_rows,
               "elapsed_s": time.perf_counter() - started,
               "limitations": ["单节点容量为一、作业不可拆分、不抢占", "所有节点与作业兼容；未模拟节点退出", "在线线性分配只优化当前一步，非全局最优", "真实耗时 oracle 仅为离线参考，不参与公平对照"]}
    np.savez_compressed(output / "trajectory.npz", **arrays)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/task_scheduling"))
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    result = run(args.output, args.seed)
    print(json.dumps({"output": str(args.output), "elapsed_s": result["elapsed_s"]}, ensure_ascii=False))
