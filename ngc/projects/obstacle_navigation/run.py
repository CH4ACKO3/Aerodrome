"""工程 01：A*、局部参考模仿学习，以及同一动力学上的闭环比较。

运行：python projects/obstacle_navigation/run.py --output outputs/navigation
只有这个文件负责场景→规划→示范→拟合→评估；没有隐藏的训练服务。
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import platform
import time

import numpy as np

from aerodrome.models.point_mass import PointMassState, step


DEFAULTS = dict(dt_s=0.1, duration_s=100.0, acceleration_limit=2.0,
                cruise_speed=1.6, velocity_gain=2.6, waypoint_radius=0.3,
                body_radius=0.25, planning_margin=0.65, grid_spacing=0.5)


def scene(seed):
    """种子决定障碍几何；训练与测试使用不同的整张地图。"""
    rng = np.random.default_rng(seed)
    centers = np.array([[5., 6.], [10., 8.], [15., 6.]])
    centers += rng.uniform(-1., 1., centers.shape)
    circles = np.column_stack([centers, rng.uniform(1.1, 1.7, 3)])
    return dict(seed=int(seed), bounds=[20., 14.], start=[1., 7.],
                goal=[19., 7.], obstacles=circles.tolist())


def segment_clearance(start, end, circles):
    """线段到圆的最小有符号距离，避免仅在离散记录帧检查碰撞。"""
    circles = np.asarray(circles)
    delta = np.asarray(end) - start
    fraction = np.clip(((circles[:, :2] - start) @ delta) /
                       max(float(delta @ delta), 1e-15), 0., 1.)
    nearest = start + fraction[:, None] * delta
    return float(np.min(np.linalg.norm(nearest - circles[:, :2], axis=1)
                        - circles[:, 2]))


def plan(scenario, config):
    """八邻域 A*，再以同一膨胀障碍执行视线简化。

    搜索每条边均检查扫掠线段，因此对角边不能穿过圆角。
    额外 planning_margin 给轨迹追踪留空间；它不是最优性的保证。
    """
    spacing = config['grid_spacing']
    bounds = np.asarray(scenario['bounds'])
    clearance = config['body_radius'] + config['planning_margin']
    start = tuple(np.rint(np.asarray(scenario['start']) / spacing).astype(int))
    goal = tuple(np.rint(np.asarray(scenario['goal']) / spacing).astype(int))
    frontier = [(0., start)]
    costs, parents = {start: 0.}, {}
    while frontier:
        _, node = heapq.heappop(frontier)
        if node == goal:
            break
        for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)):
            neighbor = (node[0] + dx, node[1] + dy)
            point = np.asarray(neighbor) * spacing
            if np.any(point < clearance) or np.any(point > bounds - clearance):
                continue
            if segment_clearance(np.asarray(node) * spacing, point,
                                 scenario['obstacles']) < clearance:
                continue
            cost = costs[node] + spacing * np.hypot(dx, dy)
            if cost < costs.get(neighbor, np.inf):
                costs[neighbor], parents[neighbor] = cost, node
                heuristic = spacing * np.linalg.norm(np.asarray(neighbor) - goal)
                heapq.heappush(frontier, (cost + heuristic, neighbor))
    else:
        raise RuntimeError('No path found for this map; save and inspect the map.')
    nodes = [goal]
    while nodes[-1] != start:
        nodes.append(parents[nodes[-1]])
    path = np.asarray(nodes[::-1]) * spacing
    simplified, index = [path[0]], 0
    while index < len(path) - 1:
        following = index + 1
        for candidate in range(len(path) - 1, index, -1):
            if segment_clearance(path[index], path[candidate],
                                 scenario['obstacles']) >= clearance:
                following = candidate
                break
        simplified.append(path[following])
        index = following
    return np.asarray(simplified)


def bounded(vector, limit):
    """限制向量的范数，而不是分别限制各分量。"""
    return vector * min(1., limit / max(float(np.linalg.norm(vector)), 1e-15))


def reference(error, speed):
    """传统局部参考：远处匀速，接近路点时连续减速。"""
    return bounded(error, speed)


def features(error, centers, width):
    """径向基仅看距离，乘位置差后保留旋转对称性。

    学习的是位置差→速度参考，而不是代替 A* 读取障碍地图。
    因此未见地图测试只检验整套组合方法，不证明策略会独立绕障。
    """
    error = np.asarray(error)
    radius = np.linalg.norm(error, axis=-1)
    radial = np.exp(-0.5 * ((radius[..., None] - centers) / width) ** 2)
    radial = np.concatenate([np.ones((*radius.shape, 1)), radial], axis=-1)
    return error[..., :, None] * radial[..., None, :]


def rollout(scenario, config, model=None):
    started = time.perf_counter()
    path = plan(scenario, config)
    planning_seconds = time.perf_counter() - started
    position, velocity = np.asarray(scenario['start'], dtype=float), np.zeros(2)
    states, actions, references, errors = [np.r_[position, velocity]], [], [], []
    index, outcome, clearance = 1, 'timeout', np.inf
    for _ in range(round(config['duration_s'] / config['dt_s'])):
        if index < len(path) - 1 and np.linalg.norm(path[index] - position) < config['waypoint_radius']:
            index += 1
        error = path[index] - position
        if model is None:
            desired_velocity = reference(error, config['cruise_speed'])
        else:
            desired_velocity = features(error, model['centers'], model['width']) @ model['weights']
            desired_velocity = bounded(desired_velocity, config['cruise_speed'])
        acceleration = bounded(config['velocity_gain'] * (desired_velocity - velocity),
                               config['acceleration_limit'])
        # 零阶保持加速度的精确双积分器更新；这里没有姿态、电机或空气动力学。
        following, next_velocity = step(PointMassState(position, velocity), acceleration,
                                        config['dt_s'])
        # 一步内实际路径是抛物线；它离端点弦的距离最多 |a| dt²/8。
        # 从弦间距扣除此界，得到保守连续间距，避免高速穿越漏检。
        curve_margin = float(np.linalg.norm(acceleration)) * config['dt_s']**2 / 8.
        clearance = min(clearance, segment_clearance(position, following,
                        scenario['obstacles']) - config['body_radius'] - curve_margin,
                        float(np.min(np.minimum(position, following) - config['body_radius'])) - curve_margin,
                        float(np.min(np.asarray(scenario['bounds']) - np.maximum(position, following) - config['body_radius'])) - curve_margin)
        actions.append(acceleration.tolist())
        references.append(desired_velocity.tolist())
        errors.append(error.tolist())
        position, velocity = following, next_velocity
        states.append(np.r_[position, velocity])
        if clearance <= 0.:
            outcome = 'collision'
            break
        if np.linalg.norm(position - scenario['goal']) < 0.2 and np.linalg.norm(velocity) < 0.2:
            outcome = 'arrived'
            break
    states, actions = np.asarray(states), np.asarray(actions)
    metrics = dict(outcome=outcome, success=outcome == 'arrived',
                   minimum_clearance_m=clearance,
                   final_error_m=float(np.linalg.norm(position - scenario['goal'])),
                   final_speed_m_s=float(np.linalg.norm(velocity)),
                   duration_s=len(actions) * config['dt_s'],
                   path_length_m=float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum()),
                   control_energy=float((actions**2).sum() * config['dt_s']),
                   planning_seconds=planning_seconds,
                   compute_seconds=time.perf_counter() - started)
    return dict(scenario=scenario, planned_path=path.tolist(), state=states.tolist(),
                acceleration=actions.tolist(), velocity_reference=references,
                waypoint_error=errors, metrics=metrics)


def train(config, seeds):
    """从传统闭环采集示范，最小二乘拟合轻量 RBF 行为克隆模型。

    训练地图内补充误差扰动以覆盖脱离示范轨迹的小范围状态；标签仍由
    相同传统参考函数给出。测试地图和测试轨迹完全不进入拟合。
    """
    errors, targets, map_seeds = [], [], []
    for seed in seeds:
        trajectory = rollout(scene(seed), config)
        observed = np.asarray(trajectory['waypoint_error'])
        rng = np.random.default_rng(seed + 1000)
        augmented = observed + rng.normal(0., 0.4, observed.shape)
        samples = np.concatenate([observed, augmented])
        errors.extend(samples)
        targets.extend([reference(error, config['cruise_speed']) for error in samples])
        map_seeds.extend([seed] * len(samples))
    errors, targets = np.asarray(errors), np.asarray(targets)
    centers = np.linspace(0., 22., 45)
    width = np.array(0.6)
    design = features(errors, centers, width).reshape(-1, len(centers) + 1)
    # 小的岭项只改善相邻径向基的数值条件；不用神经网络训练框架。
    regularization = 1e-5
    weights = np.linalg.solve(design.T @ design + regularization * np.eye(design.shape[1]),
                              design.T @ targets.reshape(-1))
    model = dict(centers=centers, width=width, weights=weights)
    return model, dict(error=errors, velocity_reference=targets,
                       map_seed=np.asarray(map_seeds)), dict(
        samples=len(errors), ridge=regularization,
        training_rmse=float(np.sqrt(np.mean((design @ weights - targets.reshape(-1))**2))))


def run(output, seed=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = DEFAULTS.copy()
    # 调参只使用 validation 地图；最终 test 地图在选择增益后才运行。
    training = list(range(seed, seed + 4))
    validation = [seed + 100, seed + 101]
    testing = [seed + 200, seed + 201, seed + 202]
    tuning = []
    for gain in (1.6, 2.6, 3.6):
        candidate = dict(config, velocity_gain=gain)
        metrics = [rollout(scene(item), candidate)['metrics'] for item in validation]
        tuning.append(dict(velocity_gain=gain,
                           successes=sum(item['success'] for item in metrics),
                           mean_duration_s=float(np.mean([item['duration_s'] for item in metrics])),
                           metrics=metrics))
    selected = min(tuning, key=lambda item: (-item['successes'], item['mean_duration_s']))
    config['velocity_gain'] = selected['velocity_gain']
    started = time.perf_counter()
    model, dataset, fit = train(config, training)
    fit['training_seconds'] = time.perf_counter() - started
    fit['device'] = f'CPU / {platform.machine()}'
    fit['numpy_version'] = np.__version__
    np.savez(output / 'model.npz', **model)
    np.savez(output / 'demonstrations.npz', **dataset)
    trials = []
    for map_seed in testing:
        for method in ('traditional', 'imitation'):
            result = rollout(scene(map_seed), config, model if method == 'imitation' else None)
            result['method'] = method
            trials.append(result)
    summary = {}
    for method in ('traditional', 'imitation'):
        selected_trials = [item['metrics'] for item in trials if item['method'] == method]
        summary[method] = dict(trials=len(selected_trials),
            successes=sum(item['success'] for item in selected_trials),
            collisions=sum(item['outcome'] == 'collision' for item in selected_trials),
            mean_duration_s=float(np.mean([item['duration_s'] for item in selected_trials])),
            worst_clearance_m=min(item['minimum_clearance_m'] for item in selected_trials))
    for name, data in [('config', dict(config=config, split=dict(train=training,
                       validation=validation, test=testing), seed=seed,
                       model='planar double integrator', version=1)),
                       ('tuning', tuning), ('training', fit), ('trajectories', trials),
                       ('summary', summary)]:
        (output / f'{name}.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    plot(trials, output / 'navigation.png')
    (output / 'report.md').write_text(
        '# 工程 01 实验记录\n\n'
        '模型：平面双积分器；所有方法共用全图 A* 和加速度限幅。\n\n'
        f'训练地图 {training}；验证地图 {validation}；测试地图 {testing}。\n\n'
        f'验证集选择 velocity_gain={config["velocity_gain"]}；完整候选记录见 tuning.json。\n\n'
        f'示范样本 {fit["samples"]}；训练 RMSE={fit["training_rmse"]:.5f} m/s。\n\n'
        '| 方法 | 到达 | 碰撞 | 平均用时/s | 最差净间距/m |\n|---|---:|---:|---:|---:|\n'
        + ''.join(f'| {name} | {data["successes"]}/{data["trials"]} | {data["collisions"]} | '
                  f'{data["mean_duration_s"]:.2f} | {data["worst_clearance_m"]:.3f} |\n'
                  for name, data in summary.items())
        + '\n行为克隆只学习局部速度参考，不学习全局障碍规划。未到达样本保留；'
          '此规模不能证明通用地图泛化，也不能代表完整旋翼或固定翼性能。\n')
    return summary


def plot(trials, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    seeds = list(dict.fromkeys(item['scenario']['seed'] for item in trials))
    figure, axes = plt.subplots(1, len(seeds), figsize=(14, 4.5), constrained_layout=True)
    for axis, map_seed in zip(np.atleast_1d(axes), seeds):
        selected = [item for item in trials if item['scenario']['seed'] == map_seed]
        for x, y, radius in selected[0]['scenario']['obstacles']:
            axis.add_patch(Circle((x, y), radius, color='gray', alpha=0.4))
        path = np.asarray(selected[0]['planned_path'])
        axis.plot(*path.T, 'k:', label='A* reference')
        for item in selected:
            states = np.asarray(item['state'])
            axis.plot(*states[:, :2].T, label=item['method'], linestyle='--' if item['method'] == 'imitation' else '-')
        axis.set(xlim=(0, 20), ylim=(0, 14), aspect='equal', title=f'Test map {map_seed}', xlabel='x / m', ylabel='y / m')
        axis.legend(fontsize=8)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/obstacle-navigation'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed), ensure_ascii=False, indent=2))
