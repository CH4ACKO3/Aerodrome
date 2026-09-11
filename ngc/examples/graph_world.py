"""Two independent entities advance in task chains, then join for reporting."""
import json
from pathlib import Path
import jax
import numpy as np
from aerodrome.composition import EntitySpec, WorldSpec, PitchAssembly, PitchInitial, build_world
from aerodrome.runners.graph_world import GraphWorldRunner
from pitch_tracking import build_case


def main():
    jax.config.update("jax_enable_x64", True)
    schedule, parameters, initial, goal = build_case()
    ids = ("aircraft_a", "aircraft_b")
    world = build_world(WorldSpec(tuple(EntitySpec(i, PitchAssembly()) for i in ids), schedule, 2))
    state = world.reset(42, {i: PitchInitial(initial.truth, initial.navigation_prior) for i in ids})
    inputs = world.pack({i: goal for i in ids})
    params = world.parameters({i: parameters for i in ids})
    reference_state, reference_trace = jax.jit(
        lambda s, p: world.rollout(s, inputs, p, steps=20))(state, params)
    completions = []
    runner = GraphWorldRunner(world, max_workers=2, chunk_ticks=7)
    final, trace = runner.run(state, inputs, params, steps=20,
                              on_complete=lambda key, _: completions.append(key))
    differences = []
    for actual, expected in zip(jax.tree.leaves(trace), jax.tree.leaves(reference_trace), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
        if np.asarray(actual).dtype.kind == "f":
            differences.append(float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))))
    assert int(final.tick) == int(reference_state.tick) == 40
    report = {
        "model": "two independent illustrative pitch entities; no performance benchmark",
        "entity_ids": ids, "physics_ticks_per_entity": 40,
        "chunk_ticks": 7, "workers": 2,
        "task_completion_order": completions,
        "max_trace_abs_difference": max(differences),
        "note": "completion order may vary; numerical time/trajectory must not",
    }
    folder = Path("artifacts/graph_world")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
