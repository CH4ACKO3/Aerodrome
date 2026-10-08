"""Run all three public scripts; compare their exports and an analytic reference."""
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_three_code_styles_reproduce_the_same_velocity_experiment(tmp_path):
    projects = Path(__file__).parents[2]/"projects/code_styles"
    records = []
    for name in ("script_style", "functional_style", "object_style"):
        folder = tmp_path/name
        run = subprocess.run([sys.executable, str(projects/f"{name}.py"), "--output", str(folder)],
                             capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stdout+run.stderr
        with np.load(folder/"trajectory.npz") as data:
            records.append({key: data[key] for key in data.files})
    for other in records[1:]:
        for key in records[0]:
            np.testing.assert_allclose(other[key], records[0][key], rtol=0, atol=1e-12)
    # The first second is at a=1; afterwards each sampled error shrinks by .98.
    # This checks the mathematical result, not only equality among programs.
    ticks = np.arange(251)
    expected_velocity = np.where(ticks <= 50, ticks*.02, 2.-.98**(ticks-50))
    np.testing.assert_allclose(records[0]["velocity_m_s"], expected_velocity, atol=1e-12)
