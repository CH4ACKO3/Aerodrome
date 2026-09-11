from hashlib import sha256
import pytest
from aerodrome.catalog import Asset, Registry
from aerodrome.pipelines import Stage, Pipeline
from aerodrome.pipelines.experiment import evaluate
from world_pitch import build_experiment


def test_asset_integrity_and_version_registry(tmp_path):
    path = tmp_path / "table.csv"
    data = b"alpha,cl\n0,0\n"
    path.write_bytes(data)
    asset = Asset("aero", "1", path, sha256(data).hexdigest(), "test fixture", "CC0")
    assert asset.read_verified() == data
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        asset.read_verified()
    registry = Registry()
    registry.register("model", "1", lambda gain: gain)
    assert registry.create("model", "1", gain=3) == 3
    with pytest.raises(ValueError, match="duplicate"):
        registry.register("model", "1", lambda: 0)
    with pytest.raises(ValueError, match="unregistered"):
        registry.create("model", "2")


def test_pipeline_orders_dependencies_and_rejects_cycles_before_execution():
    pipeline = Pipeline((Stage("report", lambda d: d["prepare"] + 1, ("prepare",)),
                         Stage("prepare", lambda _: 3)))
    assert pipeline.run()["report"] == 4
    with pytest.raises(ValueError, match="cycle"):
        Pipeline((Stage("a", lambda _: pytest.fail("must not run"), ("b",)),
                  Stage("b", lambda _: 0, ("a",))))
    with pytest.raises(ValueError, match="unknown"):
        Pipeline((Stage("a", lambda _: 0, ("missing",)),))


def test_scenario_evaluation_runs_outside_world():
    from dataclasses import replace
    world, scenario, metrics = build_experiment()
    evaluation = evaluate(world, replace(scenario, steps=4), metrics)
    assert int(evaluation.final_state.tick) == 8
    assert evaluation.trace.tick.shape == (4, 2)
    assert float(evaluation.metrics["tracking_rmse_rad"]) > 0
