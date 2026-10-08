"""Assemble and evaluate a real scenario through the experiment pipeline."""
from dataclasses import replace
import json

import numpy as np

from aerodrome.catalog import Registry
from aerodrome.pipelines import Stage, Pipeline
from aerodrome.pipelines.experiment import evaluate
from world_pitch import build_experiment


def test_registered_scenario_runs_through_pipeline_and_report(tmp_path):
    registry = Registry()
    registry.register("pitch", "1", build_experiment)

    def simulate(dependencies):
        world, scenario, metrics = dependencies["prepare"]
        return evaluate(world, replace(scenario, steps=30), metrics)

    def report(dependencies):
        result = dependencies["simulate"]
        values = {name: float(value) for name, value in result.metrics.items()}
        (tmp_path / "metrics.json").write_text(json.dumps(values))
        return values

    pipeline = Pipeline((
        Stage("report", report, ("simulate",)),
        Stage("simulate", simulate, ("prepare",)),
        Stage("prepare", lambda _: registry.create("pitch", "1", seed=42)),
    ))
    outputs = pipeline.run()
    result = outputs["simulate"]
    trace = result.trace.entities[0]
    expected = {
        "tracking_rmse_rad": np.sqrt(np.mean((trace.truth.pitch_rad - trace.reference.pitch_rad)**2)),
        "estimation_rmse_rad": np.sqrt(np.mean((trace.truth.pitch_rad - trace.navigation.estimate.pitch_rad)**2)),
    }
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert int(result.final_state.tick) == 60
    for name, value in expected.items():
        np.testing.assert_allclose(saved[name], value, atol=1e-12)
