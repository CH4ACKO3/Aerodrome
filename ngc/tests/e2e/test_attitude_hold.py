"""Run the actual PD example and check sustained recovery and saved controls."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_f16_pd_recovers_both_disturbances_and_exports_results(tmp_path):
    script = Path(__file__).parents[2]/"examples/f16_attitude_hold.py"
    result = subprocess.run([sys.executable, str(script), "--output", str(tmp_path)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path/"summary.json").read_text())
    with np.load(tmp_path/"trajectory.npz") as records:
        assert all(np.isfinite(records[key]).all() for key in records.files)
        np.testing.assert_allclose(records["time_s"][[0, -1]], [0., 20.])
        errors = np.rad2deg(records["euler_error_rad"][:, :, :2])
        np.testing.assert_allclose(errors[:, 0], [[5., 2.], [-5., -2.], [5., 2.]], atol=1e-12)
        # Both signs must recover and stay settled during the final two seconds.
        tail = records["time_s"] >= 18.
        assert np.max(np.abs(errors[:2, tail])) < .1
        assert np.max(np.abs(np.rad2deg(records["omega_body_rad_s"][:2, tail]))) < .1
        assert np.max(np.abs(records["speed_m_s"][:2, tail]-150.)) < .1
        assert np.linalg.norm(errors[0, -1]) < .01*np.linalg.norm(errors[2, -1])
        controls = records["controls"]
        assert np.all(np.abs(controls[:, :, :3]) <= np.deg2rad([25., 21.5, 30.]))
        assert np.all((controls[:, :, 4] >= 0.) & (controls[:, :, 4] <= 84500.))
        np.testing.assert_allclose(controls[2], np.broadcast_to(controls[2, 0], controls[2].shape))
    assert all(t is not None and t < 10. for t in summary["settling_time_s"][:2])
    assert (tmp_path/"control.png").stat().st_size > 0
