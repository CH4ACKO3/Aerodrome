"""Run the shipped flight experiment through simulation, evaluation and export."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_f16_level_flight_example(tmp_path):
    script = Path(__file__).resolve().parents[2] / "examples/f16_level_flight.py"
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    folder = tmp_path / "artifacts/f16_level_flight"
    summary = json.loads((folder / "summary.json").read_text())
    limits = np.asarray(summary["settling_limits"])
    assert np.all(np.abs(summary["final_errors"]) < limits)
    assert all(time is not None and time < 25 for time in summary["settling_time_s"])
    with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
        assert data["state"].shape == (6000, 3, 5)
        assert np.all(np.isfinite(data["state"]))
        np.testing.assert_allclose(data["final_state"] - data["trim_state"], summary["final_errors"], atol=1e-12)
        np.testing.assert_allclose(data["time_s"], np.arange(6000)*summary["control_dt_s"])
    assert (folder / "flight.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
