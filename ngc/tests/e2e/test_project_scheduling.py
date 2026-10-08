"""Fit duration models, solve assignments and actually clear held-out queues."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_task_scheduling_project(tmp_path):
    script = Path(__file__).resolve().parents[2] / "projects/task_scheduling/run.py"
    result = subprocess.run([sys.executable, str(script), "--output", str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    for row in summary["static_evaluation"]:
        assert row["true_total_duration"] >= row["oracle_cost"] - 1e-10
        if row["enumerated_cost"] is not None:
            np.testing.assert_allclose(row["oracle_cost"], row["enumerated_cost"], atol=1e-10)
        if row["method"] == "learned_assignment":
            assert row["prediction_rmse"] < 0.02
    with np.load(tmp_path / "trajectory.npz", allow_pickle=False) as data:
        for row in summary["online_evaluation"]:
            scene = f"online_{row['case']}_{row['repeat']}"
            key = scene + "_" + row["method"]
            nodes, start, finish = (data[key + suffix] for suffix in ("_node", "_start", "_finish"))
            assert np.all(start >= data[scene + "_arrival"] - 1e-10)
            assert np.all(np.isfinite(finish)) and np.all(finish > start)
            # Capacity one: no overlapping execution intervals at any node.
            for node in range(row["nodes"]):
                jobs = np.flatnonzero(nodes == node)
                jobs = jobs[np.argsort(start[jobs])]
                assert np.all(start[jobs[1:]] >= finish[jobs[:-1]] - 1e-10)
            np.testing.assert_allclose(np.mean(finish - data[scene + "_arrival"]), row["mean_flow_time"])
            assert row["completed_fraction"] == 1.0
