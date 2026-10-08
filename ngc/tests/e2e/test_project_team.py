"""真实入口贯通示范采集、训练、成员变化和留出规模评测。"""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_variable_team_training_and_member_events(tmp_path):
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(root / "projects/variable_team/run.py"), "--output", str(tmp_path)],
                   cwd=root, check=True, capture_output=True, text=True)
    result = json.loads((tmp_path / "metrics.json").read_text())
    # 整个留出集须完成；数量增加、加入/退出均属于同一个实际任务。
    assert len(result["evaluation"]) == 18
    assert all(r["completed"] for r in result["evaluation"])
    assert result["permutation_max_error_m"] < 1e-12
    assert {r["count"] for r in result["evaluation"]} == {3, 5, 8}
    for row in result["evaluation"]:
        with np.load(tmp_path / row["trace"]) as trace:
            active = trace["active"]
            np.testing.assert_array_equal(active.sum(-1)[[0, 120, 200]], [row["count"], row["count"] - 1, row["count"]])
            assert np.isfinite(trace["position"]).all()
            assert not np.any(trace["neighbors"] & ~active[:, None, :])
            # 独立采样解析抛物线，评价完整区间而不是绘图帧之间的直线。
            # 这里也包括退出前5.95→6秒和最后17.95→18秒的物理区间。
            dense_minimum = np.full(len(trace["time"]), np.inf)
            pairs = active[:, :, None] & active[:, None, :] & ~np.eye(active.shape[1], dtype=bool)
            for tau in np.linspace(0, 0.05, 21):
                positions = (trace["position"] + trace["velocity"] * tau +
                             0.5 * trace["action"] * tau ** 2)
                distances = np.linalg.norm(positions[:, :, None] - positions[:, None, :], axis=-1)
                dense_minimum = np.minimum(dense_minimum, np.min(np.where(pairs, distances, np.inf), axis=(1, 2)))
            assert np.all(trace["separation_lower_bound"] <= dense_minimum + 1e-12)
            # 默认输入上限下该几何界应足够紧，不可把恒零伪装为有用间距。
            assert np.max(dense_minimum - trace["separation_lower_bound"]) < 0.005
            np.testing.assert_allclose(row["minimum_separation_m"], trace["separation_lower_bound"].min())
            np.testing.assert_allclose(trace["end_time"][-1], 18.0)
            np.testing.assert_allclose(trace["end_position"][active], positions[active], atol=1e-12)
            # 新节点的加入初始化是事件跳变，不应连成之前的物理扫掠。
            joining_id = row["count"]
            slot = int(np.flatnonzero(trace["member_ids"] == joining_id)[0])
            assert np.linalg.norm(trace["end_position"][199, slot] - trace["position"][200, slot]) > 1.0

    assert (tmp_path / "training.npz").is_file()
    assert (tmp_path / "policy.npz").is_file()
