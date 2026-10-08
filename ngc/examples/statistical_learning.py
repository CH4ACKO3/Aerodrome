"""第二章统计学习短例；从 ngc/ 运行 python examples/statistical_learning.py。

只依赖 NumPy。数据均为教学设定。多数函数演示一次计算；logistic_demo 中
另有完整的训练/验证/测试流程。神经网络、卷积、注意力和图聚合采用给定权重，
不把一次前向计算称为训练成功。每个函数对应一个小节，绘图直接复用这里的结果。
"""
import argparse
import json
import numpy as np


def sigmoid(z):
    # tanh 写法与 sigmoid 数学等价，避免演示中直接计算很大的 exp(-z)。
    return .5 * (1 + np.tanh(np.asarray(z) / 2))


def fit_logistic(x, y, strength, steps=1200):
    """平均负对数似然 + strength/2 * ||w||²；最后一列是截距，不惩罚。"""
    design = np.column_stack([x, np.ones(len(x))])
    weight = np.zeros(design.shape[1])
    for _ in range(steps):
        probability = sigmoid(design @ weight)
        gradient = design.T @ (probability - y) / len(y)
        gradient[:-1] += strength * weight[:-1]
        weight -= .2 * gradient
    return weight


def binary_nll(logits, y):
    """直接从 logit 算平均对数损失，不把接近 0/1 的概率再取对数。"""
    return float(np.mean(np.logaddexp(0, logits) - y * logits))


def lda_demo():
    x = np.linspace(-3, 5, 201)
    means = np.array([0., 2.])
    density = np.exp(-.5*(x[:, None] - means)**2) / np.sqrt(2*np.pi)
    query_density = np.exp(-.5*(1.5 - means)**2)
    posterior = density[:, 1] / density.sum(axis=1)  # 两类先验都为 1/2。
    return {'x': x, 'density': density, 'posterior_class_1': posterior,
            'probability_at_x_1_5': float(query_density[1]/query_density.sum()),
            'equal_cost_boundary': float(means.mean())}


def logistic_demo(seed=7):
    """独立合成样本上的完整流程；先划分，再只用训练集估计标准化参数。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(360, 2))
    probability = sigmoid(x @ [1.5, -.8] - .2)
    y = rng.binomial(1, probability)
    indices = rng.permutation(len(x))
    train, validation, test = np.split(indices, [216, 288])
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    standardized = (x - mean) / scale
    design = np.column_stack([standardized, np.ones(len(x))])
    candidates = []
    weights = []
    for strength in [0., .03, .3]:
        weight = fit_logistic(standardized[train], y[train], strength)
        weights.append(weight)
        candidates.append(binary_nll(design[validation] @ weight, y[validation]))
    selected = int(np.argmin(candidates))
    # 超参数已由验证集确定；此处才使用测试标签，不按测试表现重新选择。
    logits = design[test] @ weights[selected]
    predicted = sigmoid(logits)
    prior = float(y[train].mean())
    prior_logit = np.log(prior/(1-prior))
    return {'seed': seed, 'split_sizes': [216, 72, 72],
            'strengths': [0., .03, .3], 'validation_nll': candidates,
            'selected_strength': [0., .03, .3][selected],
            'test_nll': binary_nll(logits, y[test]),
            'test_accuracy': float(np.mean((predicted >= .5) == y[test])),
            'test_brier': float(np.mean((predicted-y[test])**2)),
            'constant_baseline_test_nll': binary_nll(np.full(len(test), prior_logit), y[test]),
            'test_probabilities': predicted, 'test_labels': y[test]}


def regression_demo():
    x = np.array([-2., -1., 0., 1., 2.])
    y = np.array([4.1, 1.2, .1, .8, 3.8])
    design = np.column_stack([np.ones(len(x)), x, x*x])
    # 求解最小二乘，不显式计算 (X^T X)^-1。
    linear = np.linalg.lstsq(design[:, :2], y, rcond=None)[0]
    quadratic = np.linalg.lstsq(design, y, rcond=None)[0]
    return {'x': x, 'y': y, 'linear_weights': linear, 'quadratic_weights': quadratic,
            'linear_training_sse': float(np.sum((y-design[:, :2] @ linear)**2)),
            'quadratic_training_sse': float(np.sum((y-design @ quadratic)**2))}


def glm_demo():
    load = np.arange(3.)  # 负载为无量纲特征，每个计数窗口固定为 1 分钟。
    mean_count = np.exp(np.log(2) + np.log(2)*load)
    return {'load': load, 'mean_count_per_minute': mean_count,
            'zero_count_probability': np.exp(-mean_count)}


def mlp_demo():
    inputs = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    # 两个隐藏单元分别提取正、负方向差异；固定权重演示 XOR 表示能力。
    hidden = np.maximum(0, inputs @ np.array([[1., -1.], [-1., 1.]]))
    logits = 4*hidden.sum(axis=1)-2
    return {'inputs': inputs, 'hidden': hidden, 'labels': np.array([0, 1, 1, 0]),
            'probability': sigmoid(logits),
            'predicted_class': (logits >= 0).astype(int)}


def convolution_demo():
    image = np.tile([0., 0., 1., 1., 1.], (5, 1))
    kernel = np.tile([-1., 1.], (3, 1))
    response = np.empty((3, 4))
    for row in range(3):
        for column in range(4):
            # 深度学习中的“卷积”通常实现为不翻转核的互相关。
            patch = image[row:row+3, column:column+2]
            response[row, column] = np.sum(patch * kernel)
    return {'image': image, 'kernel': kernel, 'valid_response': response}


def sequence_demo():
    keys = np.array([1., 0., 2.]); values = np.array([10., 20., 30.])
    scores = np.tile(keys, (3, 1))  # 标量 query 均为 1，故 d_k=1。
    scores[np.triu_indices(3, k=1)] = -np.inf  # 当前行不能读取未来列。
    weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return {'keys': keys, 'values': values, 'causal_weights': weights,
            'outputs': weights @ values}


def neighbor_demo():
    x = np.array([0., 1., 2., 4.]); y = np.array([0., 1., 0., 4.])
    query = np.linspace(-.5, 4.5, 201)
    order = np.argsort(abs(query[:, None]-x), axis=1, kind='stable')
    predictions = {str(k): y[order[:, :k]].mean(axis=1) for k in [1, 3]}
    nearest = np.argsort(abs(x-1.8))[:3]
    return {'x': x, 'y': y, 'query': query, 'predictions': predictions,
            'three_neighbor_prediction_at_1_8': float(y[nearest].mean())}


def kernel_demo(length=1.):
    x = np.array([-2., -1., 1., 2.]); y = np.array([-.9, -.8, .8, .9])
    query = np.linspace(-4, 4, 201)
    kernel = np.exp(-.5*((x[:, None]-x)/length)**2)
    cross = np.exp(-.5*((query[:, None]-x)/length)**2)
    system = kernel + .04*np.eye(len(x))  # 高斯观测噪声方差为 .04。
    mean = cross @ np.linalg.solve(system, y)
    # 潜在函数的后验方差；新观测的方差还要额外加 .04。
    variance = 1 - np.sum(cross*np.linalg.solve(system, cross.T).T, axis=1)
    return {'x': x, 'y': y, 'query': query, 'mean': mean,
            'latent_sd': np.sqrt(np.maximum(variance, 0)), 'noise_variance': .04}


def best_stump(x, y):
    """一维回归树的一次切分；候选阈值取相邻训练输入的中点。"""
    candidates = (np.unique(x)[:-1]+np.unique(x)[1:])/2
    records = []
    for threshold in candidates:
        left = x <= threshold
        prediction = np.where(left, y[left].mean(), y[~left].mean())
        records.append((float(np.sum((y-prediction)**2)), float(threshold), prediction))
    return min(records, key=lambda record: record[0])


def tree_demo():
    x = np.arange(4.); y = np.array([0., 1., 1., 3.])
    first_sse, threshold, prediction = best_stump(x, y)
    _, residual_threshold, correction = best_stump(x, y-prediction)
    boosted = prediction + .5*correction  # 平方损失的负梯度为残差。
    return {'x': x, 'y': y, 'threshold': threshold, 'stump_prediction': prediction,
            'stump_sse': first_sse, 'residual_threshold': residual_threshold,
            'boosted_prediction': boosted, 'boosted_sse': float(np.sum((y-boosted)**2))}


def augmentation_demo():
    rotation = np.array([[0., -1.], [1., 0.]])
    position = np.array([1., 0.]); velocity = np.array([2., 0.])
    return {'position': position, 'velocity_label': velocity,
            'rotated_position': rotation @ position,
            'rotated_velocity_label': rotation @ velocity,
            'unchanged_speed_label': float(np.linalg.norm(velocity))}


def pca_demo():
    x = np.array([[-2., -.8], [-1., -.7], [1., .7], [2., .8]])
    mean = x.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(x-mean, full_matrices=False)
    direction = vt[0]
    # 特征向量正负号任意；图中统一朝右，不改变投影或重建结果。
    if direction[0] < 0: direction = -direction
    scores = (x-mean) @ direction
    reconstructed = mean + scores[:, None]*direction
    return {'x': x, 'direction': direction, 'scores': scores, 'reconstructed': reconstructed,
            'explained_variance_ratio': float(singular_values[0]**2/np.sum(singular_values**2)),
            'mean_reconstruction_squared_error': float(np.mean(np.sum((x-reconstructed)**2, axis=1)))}


def clustering_demo():
    x = np.array([0., 1., 4., 5.]); centers = np.array([0., 4.])
    history = []
    for _ in range(3):
        assignment = np.argmin((x[:, None]-centers)**2, axis=1)
        centers = np.array([x[assignment == k].mean() for k in range(2)])
        objective = float(np.sum((x-centers[assignment])**2))
        history.append({'centers': centers.copy(), 'objective': objective})
    return {'x': x, 'assignment': assignment, 'history': history}


def recommendation_demo():
    truth = np.outer([1., 2., 3.], [2., 4., 6.])  # 教学处理时间，单位秒。
    observed = np.ones((3, 3), dtype=bool); observed[1, 2] = False
    # 缺失位置为 NaN；训练只索引 observed，绝不把缺失项当 0 参与损失。
    ratings = np.where(observed, truth, np.nan)
    user = np.ones(3); item = np.ones(3)
    for _ in range(30):
        for i in range(3):
            use = observed[i]
            user[i] = np.dot(item[use], ratings[i, use]) / np.dot(item[use], item[use])
        for j in range(3):
            use = observed[:, j]
            item[j] = np.dot(user[use], ratings[use, j]) / np.dot(user[use], user[use])
    prediction = np.outer(user, item)
    return {'observed_values': np.where(observed, truth, 0), 'observed_mask': observed,
            'prediction': prediction, 'held_out_prediction': float(prediction[1, 2]),
            'held_out_truth': float(truth[1, 2])}


def graph_demo():
    adjacency = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    features = np.array([0., 2., 4.])
    with_self = adjacency + np.eye(3)
    aggregation = with_self / with_self.sum(axis=1, keepdims=True)
    history = [features]
    for _ in range(8): history.append(aggregation @ history[-1])
    return {'adjacency': adjacency, 'features': features, 'aggregation': aggregation,
            'one_round': history[1], 'history': np.array(history)}


DEMOS = {
    'lda': lda_demo, 'logistic': logistic_demo, 'regression': regression_demo,
    'glm': glm_demo, 'mlp': mlp_demo, 'convolution': convolution_demo,
    'sequence': sequence_demo, 'neighbors': neighbor_demo, 'kernels': kernel_demo,
    'trees': tree_demo, 'augmentation': augmentation_demo, 'pca': pca_demo,
    'clustering': clustering_demo, 'recommendation': recommendation_demo, 'graphs': graph_demo,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=['all', *DEMOS], default='all')
    parser.add_argument('--seed', type=int, default=7, help='只控制 logistic 的合成数据和划分')
    args = parser.parse_args()
    selected = DEMOS if args.example == 'all' else {args.example: DEMOS[args.example]}
    results = {name: function(args.seed) if name == 'logistic' else function()
               for name, function in selected.items()}
    print(json.dumps(results, ensure_ascii=False, indent=2, default=lambda value: value.tolist()))


if __name__ == '__main__':
    main()
