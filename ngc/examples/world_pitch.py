"""Run: python examples/world_pitch.py. Static ECS composition + experiment DAG."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from aerodrome.catalog import Registry
from aerodrome.composition import EntitySpec, WorldSpec, PitchAssembly, PitchInitial, build_world
from aerodrome.pipelines import Pipeline, Stage
from aerodrome.pipelines.experiment import Scenario, evaluate
from pitch_tracking import build_case


def build_experiment(seed=42):
    schedule, params, initial, goal = build_case(seed)
    registry = Registry()
    registry.register("teaching.pitch", "1", PitchAssembly)
    world = build_world(WorldSpec(
        (EntitySpec("aircraft", registry.create("teaching.pitch", "1")),),
        schedule=schedule, ticks_per_step=schedule.control_every,
    ))
    scenario = Scenario(
        "pitch-tracking", seed,
        {"aircraft": PitchInitial(initial.truth, initial.navigation_prior)},
        {"aircraft": goal}, world.parameters({"aircraft": params}), steps=500,
    )
    metrics = {
        "tracking_rmse_rad": lambda final, trace: jnp.sqrt(jnp.mean(
            (trace.entities[0].truth.pitch_rad - trace.entities[0].reference.pitch_rad)**2)),
        "estimation_rmse_rad": lambda final, trace: jnp.sqrt(jnp.mean(
            (trace.entities[0].truth.pitch_rad - trace.entities[0].navigation.estimate.pitch_rad)**2)),
    }
    return world, scenario, metrics


def main():
    jax.config.update("jax_enable_x64", True)
    world, scenario, metrics = build_experiment()

    def report(dependencies):
        evaluation = jax.device_get(dependencies["evaluate"])
        result = {
            "scenario": scenario.id, "seed": scenario.seed,
            "model": "illustrative two-state pitch; not F16",
            "backend": "jax", "jax": jax.__version__, "device": str(jax.devices()[0]),
            "dtype": str(evaluation.trace.time_s.dtype),
            "entity_ids": world.entity_ids, "decision_steps": scenario.steps,
            "ticks_per_step": world.spec.ticks_per_step,
            "physics_dt_s": world.spec.schedule.physics_dt_s,
            "final_tick": int(evaluation.final_state.tick),
            "trace_axes": ["decision_step", "physics_tick"],
            "metrics": {key: float(value) for key, value in evaluation.metrics.items()},
        }
        folder = Path("artifacts/world_pitch")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        trace = evaluation.trace.entities[0]
        np.savez(folder / "trajectory.npz", time_s=trace.time_s,
                 truth_pitch_rad=trace.truth.pitch_rad,
                 estimate_pitch_rad=trace.navigation.estimate.pitch_rad,
                 reference_pitch_rad=trace.reference.pitch_rad,
                 measurement_valid=trace.measurement.valid)
        return result

    pipeline = Pipeline((
        Stage("scenario", lambda _: scenario),
        Stage("evaluate", lambda d: evaluate(world, d["scenario"], metrics), ("scenario",)),
        Stage("report", report, ("evaluate",)),
    ))
    print(json.dumps(pipeline.run()["report"], indent=2))


if __name__ == "__main__":
    main()
