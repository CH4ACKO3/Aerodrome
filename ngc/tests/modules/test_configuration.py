"""Build configured Worlds and verify their physical behavior."""
from dataclasses import dataclass
from hashlib import sha256

import jax
import numpy as np
import pytest

from aerodrome.configuration import load_config, build_experiment, builtin_registry


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_composed_config_builds_multiple_entities(tmp_path, dtype):
    (tmp_path / "base.yaml").write_text("runtime: {steps: 3}\n")
    (tmp_path / "experiment.yaml").write_text(
        "includes: [base.yaml]\n"
        "entities:\n"
        "  - id: first\n    model: {parameters: {mass_kg: 1000.0}}\n"
        "  - id: second\n    model: {parameters: {mass_kg: 2000.0}}\n"
    )
    (tmp_path / "earth.yaml").write_text(
        "environment: {kind: constant_gravity, parameters: {gravity_ned_m_s2: [0.0, 0.0, 9.8]}}\n"
    )
    config = load_config(tmp_path / "experiment.yaml", [f"runtime.dtype={dtype}"], overlays=["earth.yaml"])
    built = build_experiment(config, base_dir=tmp_path)
    final, _ = jax.jit(lambda state: built.world.rollout(
        state, built.inputs, built.parameters, steps=config["runtime"]["steps"]
    ))(built.state)
    time = 3 * built.world.step_dt_s
    for body in final.entities:
        assert body.position_ned_m.dtype == np.dtype(dtype)
        np.testing.assert_allclose(body.position_ned_m[2], -1000 + .5*9.8*time**2, atol=1e-4, rtol=0)
        np.testing.assert_allclose(body.velocity_body_m_s[2], 9.8*time, atol=1e-6)


def test_custom_factory_uses_loaded_asset_in_simulation(tmp_path):
    payload = b"0 0 2\n"
    (tmp_path / "gravity.txt").write_bytes(payload)
    registry = builtin_registry()

    @dataclass
    class Options:
        scale: float = 1.

    def gravity(options, context):
        values = np.fromstring(context["assets"]["gravity"].decode(), sep=" ")
        return context["vector"](values*options.scale, 3, "gravity")

    registry.register("environment", "from_asset", "1", Options, gravity)
    built = build_experiment({
        "assets": {"gravity": {"path": "gravity.txt", "sha256": sha256(payload).hexdigest(),
                               "source": "test", "license": "CC0"}},
        "environment": {"kind": "from_asset", "parameters": {"scale": 3.}},
    }, base_dir=tmp_path, registry=registry)
    final, _ = jax.jit(built.world.step)(built.state, built.inputs, built.parameters)
    np.testing.assert_allclose(final.entities[0].velocity_body_m_s[2], 6*built.world.step_dt_s, atol=1e-12)
