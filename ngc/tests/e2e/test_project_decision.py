"""Actual training, held-out closed loops, matrix-game logs and replay metrics."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_dynamic_decision_project(tmp_path):
    script = Path(__file__).resolve().parents[2] / "projects/dynamic_decision/run.py"
    result = subprocess.run([sys.executable, str(script), "--output", str(tmp_path)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    np.testing.assert_allclose(summary["tracking"]["model_coefficients"], [-0.4, -0.18, 1.0], atol=0.005)
    with np.load(tmp_path / "trajectory.npz", allow_pickle=False) as data:
        for row in summary["tracking"]["evaluation"]:
            prefix = f"tracking_{row['case']}_{row['method']}"
            state, ref = data[prefix + "_state"], data[prefix + "_reference"]
            rmse = np.sqrt(np.mean((state[:, 0] - ref)**2))
            np.testing.assert_allclose(rmse, row["rmse"], atol=1e-12)
            assert rmse < 0.2
            assert np.max(np.abs(data[prefix + "_command"])) <= 6
        # The matrix game has an independent closed-form cyclic payoff identity.
        for row in summary["game"]["evaluation"]:
            prefix = f"game_{row['case']}_{row['repeat']}_{row['method']}"
            delta = (data[prefix + "_actions"] - data[prefix + "_opponent"]) % 3
            expected = np.where(delta == 1, 1, np.where(delta == 2, -1, 0))
            np.testing.assert_array_equal(data[prefix + "_reward"], expected)
            np.testing.assert_allclose(expected.mean(), row["mean_reward"])
        learned = [row for row in summary["game"]["evaluation"] if row["method"] == "learned_transition"]
        assert np.mean([row["mean_reward"] for row in learned if row["case"] == 1]) > 0.7
        switched = [row for row in learned if row["switch"]]
        assert np.mean([row["first_half_reward"] - row["second_half_reward"] for row in switched]) > 0.5
