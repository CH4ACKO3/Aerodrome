"""Run the public CLI and inspect the experiment files it produces."""
import json
import subprocess
import sys

import numpy as np
import pytest
import yaml


def run_cli(directory, config, *args):
    path = directory / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    return subprocess.run(
        [sys.executable, "-m", "aerodrome.configuration.cli", "--config", str(path), *args],
        cwd=directory, text=True, capture_output=True, timeout=120,
    )


@pytest.mark.parametrize("trace", [True, False])
def test_config_sweep_simulates_and_saves_results(tmp_path, trace):
    config = {
        "runtime": {"steps": 5, "chunk_steps": 2, "trace": trace},
        "renderer": {"kind": "headless", "parameters": {"every_steps": 2}},
        "entities": [{"id": "body", "model": {"kind": "rigid_body", "parameters": {
            "mass_kg": 2., "position_ned_m": [0., 0., 0.],
            "velocity_body_m_s": [0., 0., 0.], "omega_body_rad_s": [0., 0., 0.],
            "force_body_N": [4., 0., 0.],
        }}}],
    }
    result = run_cli(tmp_path, config, "--sweep", "seed=[1,2]")
    assert result.returncode == 0, result.stdout + result.stderr
    runs = list((tmp_path / "artifacts/config_runs").iterdir())
    assert {json.loads((run / "resolved.json").read_text())["seed"] for run in runs} == {1, 2}
    assert len(runs) == 2
    for run in runs:
        status = json.loads((run / "status.json").read_text())
        assert status["status"] == "complete"
        assert status["tick"] == 10 and status["rendered_frames"] == 2
        paths = json.loads((run / "final.paths.json").read_text())
        with np.load(run / "final.npz", allow_pickle=False) as saved:
            state = {path: saved[f"leaf_{i}"] for i, path in enumerate(paths)}
        np.testing.assert_allclose(state[".entities[0].velocity_body_m_s"], [.2, 0., 0.], atol=1e-12)
        np.testing.assert_allclose(state[".entities[0].position_ned_m"], [.01, 0., 0.], atol=1e-12)
        chunks = sorted(run.glob("trace_*.npz"))
        if trace:
            ticks = []
            for chunk in chunks:
                paths = json.loads(chunk.with_suffix(".paths.json").read_text())
                with np.load(chunk, allow_pickle=False) as saved:
                    ticks.extend(saved[f"leaf_{paths.index('.tick')}"].ravel())
            np.testing.assert_array_equal(ticks, np.arange(10))
        else:
            assert not chunks


def test_missing_experiment_asset_reports_failure(tmp_path):
    result = run_cli(tmp_path, {"assets": {"table": {
        "path": "missing.csv", "sha256": "0" * 64, "source": "test", "license": "CC0",
    }}})
    assert result.returncode != 0
    run = next((tmp_path / "artifacts/config_runs").iterdir())
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "failed"
    assert not (run / "final.npz").exists()


def test_f16_six_dof_trim_flight_and_rendering(tmp_path):
    from pathlib import Path
    from aerodrome.configuration import api
    presets = Path(api.__file__).parent / "conf"
    config = api.load_config(presets/"experiment.yaml", overlays=[
        "scenario/f16_6dof.yaml", "environment/earth.yaml", "renderer/headless.yaml"],
        overrides=["runtime.steps=100", "renderer.parameters.every_steps=20"])
    result = run_cli(tmp_path, config)
    assert result.returncode == 0, result.stdout + result.stderr
    run = next((tmp_path / "artifacts/config_runs").iterdir())
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "complete" and status["rendered_frames"] == 5
    paths = json.loads((run / "final.paths.json").read_text())
    with np.load(run / "final.npz", allow_pickle=False) as saved:
        state = {path: saved[f"leaf_{i}"] for i, path in enumerate(paths)}
    position = state[".entities[0].position_ned_m"]
    np.testing.assert_allclose(position[2], -3000., atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(position[:2]), 300., atol=1e-8)
    np.testing.assert_allclose(state[".entities[0].omega_body_rad_s"], 0., atol=1e-10)
    assert "builtin/f16_6dof" in json.loads((run / "assets.json").read_text())
