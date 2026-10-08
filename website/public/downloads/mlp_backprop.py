"""两输入、两隐藏单元、单输出网络的一次前向、反向与梯度下降。

从 ngc/ 运行：python examples/mlp_backprop.py
只需要 NumPy；输入和目标已无量纲化，列向量约定与教材 2.3 一致。
本例使用单样本平方损失与 ReLU，不是完整的数据集训练实验。
"""
import json
import numpy as np


def forward(parameters, x, target):
    """单样本在 NumPy 中存为一维数组，数学上按列向量理解。

    W1 为 (2,2)，b1 为 (2,)，W2 为 (1,2)，b2 为 (1,)；
    预测和损失为标量。返回中间量，便于观察及重复利用计算结果。
    """
    z = parameters['W1'] @ x + parameters['b1']
    hidden = np.maximum(0., z)
    prediction = float((parameters['W2'] @ hidden + parameters['b2'])[0])
    residual = prediction - target
    return dict(z=z, hidden=hidden, prediction=prediction,
                residual=residual, loss=0.5*residual**2)


def backward(parameters, x, cache):
    """链式法则：输出误差经旧 W2 向后传，外积生成逐参数梯度。

    ReLU 在零点不可微，这里取导数 0；固定算例的两个 z 均严格为正。
    np.outer 实现列向量与行向量的外积；星号是逐元素乘法。
    """
    delta2 = np.array([cache['residual']])
    hidden_gradient = parameters['W2'].T @ delta2
    delta1 = hidden_gradient * (cache['z'] > 0)
    gradients = dict(W1=np.outer(delta1, x), b1=delta1,
                     W2=np.outer(delta2, cache['hidden']), b2=delta2)
    input_gradient = parameters['W1'].T @ delta1
    return gradients, input_gradient


def finite_difference(parameters, x, target, step=1e-6):
    """只用前向损失、逐个扰动参数，独立近似导数以核对反传。

    中心差分需要两次前向计算/参数，适合这个小例子的教学核对，
    不适合代替大网络的反向传播。每次计算从原参数独立复制。
    """
    numerical = {}
    for name, value in parameters.items():
        numerical[name] = np.empty_like(value)
        for index in np.ndindex(value.shape):
            plus = {key: item.copy() for key, item in parameters.items()}
            minus = {key: item.copy() for key, item in parameters.items()}
            plus[name][index] += step
            minus[name][index] -= step
            numerical[name][index] = (
                forward(plus, x, target)['loss'] - forward(minus, x, target)['loss']
            ) / (2*step)
    return numerical


def demo():
    x = np.array([1., 2.])
    target = 1.
    parameters = dict(W1=np.array([[.1, .2], [-.3, .4]]), b1=np.array([.1, .2]),
                      W2=np.array([[.5, -.2]]), b2=np.array([.1]))
    before = forward(parameters, x, target)
    gradients, input_gradient = backward(parameters, x, before)
    numerical = finite_difference(parameters, x, target)
    error = max(float(np.max(np.abs(gradients[k]-numerical[k]))) for k in gradients)
    # 先求完全部梯度，再同时更新；不能用已更新的 W2 来算 W1 的梯度。
    learning_rate = .1
    updated = {name: value-learning_rate*gradients[name] for name, value in parameters.items()}
    after = forward(updated, x, target)
    return dict(input=x, target=target, parameters=parameters, before=before,
                gradients=gradients, input_gradient=input_gradient,
                numerical_gradients=numerical, max_gradient_error=error,
                learning_rate=learning_rate, updated_parameters=updated, after=after)


if __name__ == '__main__':
    print(json.dumps(demo(), ensure_ascii=False, indent=2, default=lambda value: value.tolist()))
