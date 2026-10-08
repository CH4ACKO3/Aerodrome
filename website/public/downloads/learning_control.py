"""第二章的采样 Q 学习与比例控制调参；只依赖 NumPy。

从 ngc/ 运行：python examples/learning_control.py --example all
Q 学习使用可重置的三站点模拟器；控制例使用一维速度积分器。
两例分别解释价值更新和控制器参数选择，不代表完整飞行器模型。
"""
import argparse
import json
import numpy as np


def station_step(state, action):
    """状态为站点编号；动作 0 等待、1 前进。到达 2 后终止。"""
    next_state = state + action
    return next_state, 1.0, next_state == 2


def q_learning_demo(seed=7):
    rng = np.random.default_rng(seed)
    gamma = 0.9
    q = np.zeros((3, 2))
    visits = np.zeros((2, 2), dtype=int)
    samples, estimates = [0], [q.min(axis=1).copy()]
    # 本例允许重置到任一非终止状态，均匀采样全部状态—动作对。
    # 更新器只接收 (s,a,c,s',done)，不读取模拟器的转移公式。
    for n in range(1, 2001):
        state, action = rng.integers(0, 2, size=2)
        next_state, cost, done = station_step(state, action)
        visits[state, action] += 1
        alpha = visits[state, action] ** -0.6
        target = cost if done else cost + gamma * q[next_state].min()
        q[state, action] += alpha * (target - q[state, action])
        if n <= 100 or n % 20 == 0:
            samples.append(n)
            estimates.append(q.min(axis=1).copy())
    policy = q[:2].argmin(axis=1)
    # 用已知的教学环境独立评价学到的策略，不拿 Q 估计冒充真实代价。
    transition = np.zeros((2, 2))
    for state, action in enumerate(policy):
        next_state, _, done = station_step(state, action)
        if not done:
            transition[state, next_state] = 1
    policy_values = np.linalg.solve(np.eye(2) - gamma * transition, np.ones(2))
    return dict(seed=seed, gamma=gamma, q=q, visits=visits, policy=policy,
                policy_values=policy_values, optimal_values=np.array([1.9, 1., 0.]),
                samples=samples, estimates=np.array(estimates))


def velocity_rollout(gain, initial_error):
    """固定速度参考，dt=0.1 s，加速度保持且限幅 ±3 m/s²。

    e=v_ref-v，a=clip(K e)。误差递推对这个积分器模型是精确的。
    代价使用 e_ref=1 m/s、a_ref=1 m/s²、t_ref=1 s 无量纲化。
    """
    dt = 0.1
    errors, actions = [float(initial_error)], []
    for _ in range(50):
        acceleration = float(np.clip(gain * errors[-1], -3., 3.))
        actions.append(acceleration)
        errors.append(errors[-1] - dt * acceleration)
    errors, actions = np.array(errors), np.array(actions)
    objective = dt * np.sum(errors[:-1]**2 + 0.2 * actions**2) + errors[-1]**2
    return dict(time=np.arange(51)*dt, error=errors, acceleration=actions,
                objective=float(objective))


def control_demo():
    # 少量候选直接搜索即可。只在训练工况中保留最好的三个供验证选择。
    gains = np.array([.2, .5, 1., 2., 4., 8., 12.])  # 单位 s^-1
    training_errors, validation_errors, test_errors = [1., 3.], [2., 4.], [-4., -1.5, 5.]
    def costs(candidates, initial_errors):
        return np.array([np.mean([velocity_rollout(k, e)['objective']
                                 for e in initial_errors]) for k in candidates])
    train_costs = costs(gains, training_errors)
    shortlist = np.argsort(train_costs)[:3]
    validation_costs = costs(gains[shortlist], validation_errors)
    gain = float(gains[shortlist[np.argmin(validation_costs)]])
    # 选定增益后才在留出工况运行，与预先固定的 K=0.5 比较。
    test_costs = [velocity_rollout(gain, e)['objective'] for e in test_errors]
    baseline_costs = [velocity_rollout(.5, e)['objective'] for e in test_errors]
    return dict(gains=gains, training_errors=training_errors, train_costs=train_costs,
                validation_errors=validation_errors, shortlist=gains[shortlist],
                validation_costs=validation_costs, selected_gain=gain,
                test_errors=test_errors, test_costs=test_costs, baseline_costs=baseline_costs,
                selected_trace=velocity_rollout(gain, 5.), baseline_trace=velocity_rollout(.5, 5.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', choices=['q-learning', 'control', 'all'], default='all')
    parser.add_argument('--seed', type=int, default=7, help='只影响 Q 学习采样顺序')
    args = parser.parse_args()
    results = {}
    if args.example in ('q-learning', 'all'):
        results['q-learning'] = q_learning_demo(args.seed)
    if args.example in ('control', 'all'):
        results['control'] = control_demo()
    print(json.dumps(results, ensure_ascii=False, indent=2,
                     default=lambda value: value.tolist()))


if __name__ == '__main__':
    main()
