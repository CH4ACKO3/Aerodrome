"""Port-wired 1D speed tracking; automatic fusion across modules and ticks.

Perfect speed observation, proportional throttle, sampled static engine map,
Euler body dynamics. This is an architecture example, not a calibrated aircraft.
"""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from aerodrome.adapters.external import Port
from aerodrome.composition.module_graph import (
    Module, ModuleGraph, Wire, InputBinding, ModuleValue, ModuleGraphState,
)
from aerodrome.runners.module_compiler import CompiledModuleGraph


def speed(name):
    return Port(name, "m/s", (), "inertial", quantity="longitudinal_velocity")


def throttle(name):
    return Port(name, "1", (), "scalar", quantity="throttle_fraction")


def thrust(name):
    return Port(name, "N", (), "body", quantity="axial_force", reference_point="center_of_mass")


def build_case(backend="jax", ticks=16):
    host_calls = []
    def observe(state, inputs, p, context):
        return ModuleValue(state, {"estimate": inputs["speed"]})
    def control(state, inputs, p, context):
        command = jnp.clip(p["gain"]*(inputs["target"]-inputs["estimate"]), 0., 1.)
        return ModuleValue(state, {"throttle": command})
    def engine(state, inputs, p, context):
        if backend == "host":
            host_calls.append(int(context.tick))
            force = np.asarray(inputs["throttle"]) * np.asarray(p["max_thrust"])
        else:
            force = inputs["throttle"] * p["max_thrust"]
        return ModuleValue(state, {"thrust": force})
    def body(state, inputs, p, context):
        following = state + context.physics_dt_s*(inputs["thrust"]-p["drag"]*state)/p["mass"]
        return ModuleValue(following, {"next_speed": following})
    modules = (
        Module("observation", (speed("speed"),), (speed("estimate"),), observe, equation="perfect_speed"),
        Module("control", (speed("estimate"), speed("target")), (throttle("throttle"),), control,
               equation="throttle=clip(K*(target-estimate),0,1)"),
        Module("engine", (throttle("throttle"),), (thrust("thrust"),), engine, backend, every=4,
               equation="T=throttle*Tmax; ZOH for 4 ticks"),
        Module("body", (thrust("thrust"),), (speed("next_speed"),), body,
               equation="v_next=v+dt*(T-drag*v)/mass"),
    )
    graph = ModuleGraph(modules, (
        # body publishes v_(k+1) in event k. Next tick's observation reads it.
        Wire(("body", "next_speed"), ("observation", "speed"), delay=1),
        Wire(("observation", "estimate"), ("control", "estimate")),
        Wire(("control", "throttle"), ("engine", "throttle")),
        Wire(("engine", "thrust"), ("body", "thrust")),
    ), inputs=(speed("target"),), bindings=(InputBinding("target", ("control", "target")),))
    zero = jnp.asarray(0.)
    initial = ModuleGraphState(jnp.asarray(0, jnp.int32), {
        "observation": ModuleValue((), {"estimate": zero}),
        "control": ModuleValue((), {"throttle": zero}),
        "engine": ModuleValue((), {"thrust": zero}),
        "body": ModuleValue(zero, {"next_speed": zero}),
    })
    parameters = {"observation": {}, "control": {"gain": jnp.asarray(0.1)},
                  "engine": {"max_thrust": jnp.asarray(1000.)},
                  "body": {"mass": jnp.asarray(100.), "drag": jnp.asarray(10.)}}
    return graph, initial, {"target": jnp.full(ticks, 12.)}, parameters, host_calls


def main():
    jax.config.update("jax_enable_x64", True)
    graph, state, inputs, parameters, _ = build_case()
    pure = CompiledModuleGraph(graph, ticks=16, physics_dt_s=0.02, chunk_ticks=32)
    pure_result = pure.run(state, inputs, parameters)
    # Actually lower/compile the full 16-tick native scan into one executable.
    executable = pure.native.lower(state, inputs, parameters).compile()
    reference = jax.block_until_ready(executable(state, inputs, parameters))
    hybrid_graph, hybrid_state, hybrid_inputs, hybrid_params, calls = build_case("host")
    hybrid = CompiledModuleGraph(hybrid_graph, ticks=16, physics_dt_s=0.02, chunk_ticks=32)
    hybrid_result = hybrid.run(hybrid_state, hybrid_inputs, hybrid_params)
    for result in (pure_result, hybrid_result):
        for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(reference), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    report = {
        "example": "1D speed tracking; host engine is an in-process NumPy fixture, not MATLAB",
        "ticks": 16, "dt_s": 0.02, "original_module_events": len(pure.events),
        "pure_regions": len(pure.regions), "hybrid_regions": len(hybrid.regions),
        "host_call_ticks": calls, "native_scan_compiled_and_checked": True,
        "final_speed_m_s": float(reference[0].modules["body"].state),
        "pure_plan": pure.describe(), "hybrid_plan": hybrid.describe(),
    }
    folder = Path("artifacts/compiled_modules")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("plan")}, indent=2))


if __name__ == "__main__":
    main()
