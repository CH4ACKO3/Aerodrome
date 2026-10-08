"""数学工具的短算例：测量、拟合与决策，不依赖完整仿真器。

从 ngc/ 执行：python examples/math_tools.py --seed 7 --repeats 2000
数值均为教学设定；输出 JSON，方便比较修改噪声或正则强度后的结果。
"""

import argparse
import json

import numpy as np


def measurement_example(seed=7, repeats=2000):
    """独立、无偏的两次位置测量：方差越小，权重越大。"""
    readings = np.array([9.8, 10.2])  # m，两传感器观测同一静止位置。
    variances = np.array([0.2, 0.4]) ** 2  # m²，假设已知的测量方差。
    precision = 1 / variances
    weights = precision / precision.sum()
    # 这里融合的是独立误差；共享漂移时不能继续使用这个方差公式。
    covariance = np.array([[4.0, 1.2], [1.2, 1.0]])  # 两维位置误差，m²。
    transform = np.array([[1.0, 1.0], [0.0, 1.0]])  # 剪切坐标，不是旋转。
    # 相同模型的有限样本协方差会波动，解析传播值没有这种抽样误差。
    rng = np.random.default_rng(seed)
    errors = rng.multivariate_normal(np.zeros(2), covariance, size=repeats)
    transformed_errors = errors @ transform.T  # 每行一个样本，故右乘转置。
    # 两次等精度静止观测的最近模型点；残差与列空间正交。
    observations = np.array([1.0, 2.0])
    design = np.ones((2, 1))
    fitted = design @ np.linalg.lstsq(design, observations, rcond=None)[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening = np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return {
        "weights": weights.tolist(),
        "fused_position_m": float(weights @ readings),
        "fused_variance_m2": float(1 / precision.sum()),
        "input_covariance_m2": covariance.tolist(),
        "output_covariance_m2": (transform @ covariance @ transform.T).tolist(),
        "sample_output_covariance_m2": np.cov(transformed_errors, rowvar=False).tolist(),
        "whitened_covariance": (whitening @ covariance @ whitening.T).tolist(),
        "projection_m": fitted.tolist(),
        "projection_residual_m": (observations - fitted).tolist(),
    }


def fit_line(ridge_strength=2.0):
    """只惩罚速度偏离参考值，截距不受惩罚。

    目标为位置残差平方和 + λ(v-v_ref)²。λ 的单位是 s²，
    两项才同为 m²。先消去截距，再求一个标量速度，便于手算核对。
    """
    time = np.array([0.0, 1.0, 2.0])
    positions = np.array([1.1, 2.9, 5.2])
    centered_time = time - time.mean()
    centered_positions = positions - positions.mean()
    speed_reference = 2.0
    results = {}
    for name, strength in [("ordinary", 0.0), ("regularized", ridge_strength)]:
        speed = (centered_time @ centered_positions + strength * speed_reference) / (
            centered_time @ centered_time + strength
        )
        initial_position = positions.mean() - speed * time.mean()
        residual = positions - (initial_position + speed * time)
        results[name] = {
            "initial_position_m": float(initial_position),
            "speed_m_s": float(speed),
            "training_sse_m2": float(residual @ residual),
        }
    return results


def repeated_predictions(seed=7, repeats=2000, sigma=0.2):
    """每行是一条新生成的三点轨迹；真值 p(t)=1+2t。

    全部噪声独立且为 N(0,σ²)。每轮重新拟合后，在 t=3 s 外推，
    再独立产生一条未来观测。返回的两个误差分别对应均值预测和新读数。
    """
    rng = np.random.default_rng(seed)
    time = np.array([0.0, 1.0, 2.0])
    design = np.column_stack([np.ones(3), time])
    readings = 1 + 2 * time + rng.normal(0, sigma, (repeats, 3))
    # 一次求解多个右端项；每列对应一次独立实验，不形成正规方程求逆。
    parameters = np.linalg.lstsq(design, readings.T, rcond=None)[0].T
    predicted_mean = parameters @ np.array([1.0, 3.0])
    new_readings = 7 + rng.normal(0, sigma, repeats)
    mean_error = predicted_mean - 7
    new_reading_error = new_readings - predicted_mean
    centered_time = time - time.mean()
    mean_variance = sigma**2 * (1 / len(time) + (3 - time.mean()) ** 2 / (centered_time @ centered_time))
    summary = {
        "seed": seed,
        "repeats": repeats,
        "observation_sigma_m": sigma,
        "speed_sd_empirical_m_s": float(parameters[:, 1].std(ddof=1)),
        "speed_sd_theory_m_s": float(sigma / np.sqrt(centered_time @ centered_time)),
        "mean_prediction_sd_empirical_m": float(mean_error.std(ddof=1)),
        "mean_prediction_sd_theory_m": float(np.sqrt(mean_variance)),
        "new_reading_error_sd_empirical_m": float(new_reading_error.std(ddof=1)),
        "new_reading_error_sd_theory_m": float(np.sqrt(mean_variance + sigma**2)),
    }
    return mean_error, new_reading_error, summary


def decision_examples():
    """评价真实后果：预测误差更小，未必选出代价更低的行动。"""
    durations = np.array([2.0, 2.0, 8.0])  # 三个等可能的处理时间，s。
    estimates = {}
    for name, value in [("mean", durations.mean()), ("median", np.median(durations))]:
        estimates[name] = {
            "prediction_s": float(value),
            "expected_squared_loss_s2": float(np.mean((durations - value) ** 2)),
            "expected_absolute_loss_s": float(np.mean(abs(durations - value))),
        }
    truth = np.array([[2.0, 3.0], [3.0, 5.0]])
    models = {"A": np.array([[2.0, 3.6], [3.6, 5.0]]), "B": truth + 2}
    assignments = [(0, 1), (1, 0)]  # 第 i 个元素表示节点 i 接收的任务编号。
    results = {}
    for name, prediction in models.items():
        costs = [float(prediction[0, order[0]] + prediction[1, order[1]]) for order in assignments]
        selected = assignments[int(np.argmin(costs))]
        results[name] = {
            "prediction_rmse_s": float(np.sqrt(np.mean((prediction - truth) ** 2))),
            "tasks_for_nodes": list(selected),
            # 目标是两项时间之和，不是并行任务的最大完成时间。
            "actual_total_time_s": float(truth[0, selected[0]] + truth[1, selected[1]]),
        }
    return {"continuous_prediction": estimates, "assignment": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--sigma", type=float, default=0.2, help="重复测量的标准差，单位 m")
    parser.add_argument("--ridge-strength", type=float, default=2.0, help="速度正则权重，单位 s²")
    args = parser.parse_args()
    _, _, summary = repeated_predictions(args.seed, args.repeats, args.sigma)
    print(json.dumps({"measurement": measurement_example(args.seed, args.repeats), "line_fit": fit_line(args.ridge_strength),
                      "repeated_fits": summary, "decision": decision_examples()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
