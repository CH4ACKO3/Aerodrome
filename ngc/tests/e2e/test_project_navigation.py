"""真实训练→留出地图闭环→产物检查，不锁定内部规划或回归实现。"""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_navigation_training_and_held_out_closed_loop(tmp_path):
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, 'projects/obstacle_navigation/run.py',
                    '--output', str(tmp_path)], cwd=root, check=True,
                   capture_output=True, text=True, timeout=120)
    saved = json.loads((tmp_path / 'config.json').read_text())
    config, split = saved['config'], saved['split']
    assert set(split['train']).isdisjoint(split['test'])
    assert set(split['validation']).isdisjoint(split['test'])
    training = json.loads((tmp_path / 'training.json').read_text())
    assert training['samples'] > 100
    assert training['training_rmse'] < 0.04
    with np.load(tmp_path / 'demonstrations.npz') as dataset:
        assert set(dataset['map_seed']) == set(split['train'])
        assert dataset['error'].shape == dataset['velocity_reference'].shape
    with np.load(tmp_path / 'model.npz') as model:
        assert np.isfinite(model['weights']).all()
        assert np.linalg.norm(model['weights']) > 0

    trajectories = json.loads((tmp_path / 'trajectories.json').read_text())
    assert len(trajectories) == 2 * len(split['test'])
    for trial in trajectories:
        states, controls = np.asarray(trial['state']), np.asarray(trial['acceleration'])
        assert trial['scenario']['seed'] in split['test']
        assert trial['metrics']['success']
        assert np.linalg.norm(states[-1, :2] - trial['scenario']['goal']) < 0.2
        assert np.linalg.norm(states[-1, 2:]) < 0.2
        assert np.max(np.linalg.norm(controls, axis=1)) <= config['acceleration_limit'] + 1e-12
        # 独立地在每个保持区间细采样实际抛物线，检查整次任务的几何结果。
        times = np.linspace(0., config['dt_s'], 11)[None, :, None]
        points = states[:-1, None, :2] + states[:-1, None, 2:] * times + 0.5 * controls[:, None, :] * times**2
        circles = np.asarray(trial['scenario']['obstacles'])
        distance = np.linalg.norm(points[:, :, None, :] - circles[None, None, :, :2], axis=-1)
        assert np.min(distance - circles[None, None, :, 2] - config['body_radius']) > 0.
        assert np.min(points) > config['body_radius']
        assert np.all(points < np.asarray(trial['scenario']['bounds']) - config['body_radius'])
    assert (tmp_path / 'navigation.png').stat().st_size > 1000
    assert (tmp_path / 'report.md').exists()
