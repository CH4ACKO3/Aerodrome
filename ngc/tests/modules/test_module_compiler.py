import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aerodrome.adapters.external import Port
from aerodrome.composition.module_graph import (
    Module, ModuleGraph, Wire, InputBinding, ModuleValue, ModuleGraphState,
)
from aerodrome.runners.module_compiler import CompiledModuleGraph


def port(name):
    return Port(name, "1", (), "scalar", quantity="test_scalar")


def value(state=0.):
    return ModuleValue(jnp.asarray(state), {"y": jnp.asarray(state)})


def accumulator(name="acc", backend="jax", every=1):
    def step(state, inputs, p, context):
        following = state + p["gain"] * inputs["u"]
        return ModuleValue(following, {"y": following})
    return Module(name, (port("u"),), (port("y"),), step, backend, every)


def test_connected_modules_match_reference_rollout_and_gradient():
    graph = ModuleGraph((accumulator("a"), accumulator("b")),
                        (Wire(("a", "y"), ("b", "u")),),
                        inputs=(port("drive"),), bindings=(InputBinding("drive", ("a", "u")),))
    program = CompiledModuleGraph(graph, ticks=6, physics_dt_s=0.1)
    state = ModuleGraphState(jnp.asarray(0), {"a": value(), "b": value()})
    parameters = {"a": {"gain": jnp.asarray(1.)}, "b": {"gain": jnp.asarray(2.)}}
    inputs = {"drive": jnp.arange(1., 7.)}
    final, trace = program.run(state, inputs, parameters)
    a = np.cumsum(np.arange(1., 7.))
    b = np.cumsum(2*a)
    np.testing.assert_allclose(trace.outputs["a"]["y"], a)
    np.testing.assert_allclose(trace.outputs["b"]["y"], b)
    assert float(final.modules["b"].state) == b[-1]
    native = program.native(state, inputs, parameters)
    for x, y in zip(jax.tree.leaves((final, trace)), jax.tree.leaves(native), strict=True):
        np.testing.assert_allclose(x, y)
    def loss(gain):
        p = {**parameters, "b": {"gain": gain}}
        return program.native(state, inputs, p)[0].modules["b"].state
    np.testing.assert_allclose(jax.grad(loss)(jnp.asarray(2.)), np.sum(a))


def test_explicit_previous_tick_feedback_uses_initial_outputs():
    graph = ModuleGraph((accumulator("a"), accumulator("b")),
                        (Wire(("b", "y"), ("a", "u"), delay=1),
                         Wire(("a", "y"), ("b", "u"))))
    state = ModuleGraphState(jnp.asarray(0), {"a": value(1.), "b": value(2.)})
    p = {name: {"gain": jnp.asarray(1.)} for name in ("a", "b")}
    _, trace = CompiledModuleGraph(graph, ticks=3, physics_dt_s=0.1).run(state, {}, p)
    # a_k=a_previous+b_(k-1); b_k=b_previous+a_k
    np.testing.assert_array_equal(trace.outputs["a"]["y"], [3., 8., 21.])
    np.testing.assert_array_equal(trace.outputs["b"]["y"], [5., 13., 34.])


def test_host_sampling_holds_outputs_between_updates():
    calls = []
    def host(state, inputs, p, context):
        calls.append(int(context.tick))
        following = np.asarray(state) + 1.
        return ModuleValue(following, {"y": following})
    graph = ModuleGraph((Module("host", (), (port("y"),), host, "host", 3), accumulator("body")),
                        (Wire(("host", "y"), ("body", "u")),))
    program = CompiledModuleGraph(graph, ticks=7, physics_dt_s=0.1)
    state = ModuleGraphState(jnp.asarray(0), {"host": value(), "body": value()})
    final, trace = program.run(state, {}, {"host": {}, "body": {"gain": jnp.asarray(1.)}})
    assert calls == [0, 3, 6]
    np.testing.assert_array_equal(trace.outputs["host"]["y"], [1., 1., 1., 2., 2., 2., 3.])
    np.testing.assert_array_equal(trace.outputs["body"]["y"], [1., 2., 3., 5., 7., 9., 12.])
    assert int(final.tick) == 7 and program.native is None


def test_jax_host_jax_feedback_preserves_update_order():
    def combine(state, inputs, p, context):
        following = state + inputs["u"] + inputs["direct"]
        return ModuleValue(following, {"y": following})
    b = Module("b", (port("u"), port("direct")), (port("y"),), combine)
    # The direct A->B edge would invite fusion, but A->host->B forbids it.
    graph = ModuleGraph((accumulator("a"), accumulator("host", "host"), b),
                        (Wire(("a", "y"), ("host", "u")), Wire(("host", "y"), ("b", "u")),
                         Wire(("a", "y"), ("b", "direct"))),
                        inputs=(port("x"),), bindings=(InputBinding("x", ("a", "u")),))
    program = CompiledModuleGraph(graph, ticks=3, physics_dt_s=0.1)
    state = ModuleGraphState(jnp.asarray(0), {name: value() for name in graph.modules})
    params = {name: {"gain": jnp.asarray(1.)} for name in graph.modules}
    _, trace = program.run(state, {"x": jnp.ones(3)}, params)
    np.testing.assert_array_equal(trace.outputs["b"]["y"], [2., 7., 16.])


@pytest.mark.parametrize("chunk_ticks", [1, 3, 10])
def test_independent_modules_match_scan_across_chunks(chunk_ticks):
    graph = ModuleGraph((accumulator("a"), accumulator("b")), inputs=(port("x"),),
                        bindings=(InputBinding("x", ("a", "u")), InputBinding("x", ("b", "u"))))
    state = ModuleGraphState(jnp.asarray(0), {"a": value(), "b": value()})
    parameters = {"a": {"gain": jnp.asarray(1.)}, "b": {"gain": jnp.asarray(2.)}}
    inputs = {"x": jnp.arange(1., 8.)}
    program = CompiledModuleGraph(graph, ticks=7, physics_dt_s=.1, chunk_ticks=chunk_ticks)
    actual = program.run(state, inputs, parameters)
    expected = program.native(state, inputs, parameters)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(actual[1].outputs["a"]["y"], np.cumsum(inputs["x"]))
    np.testing.assert_allclose(actual[1].outputs["b"]["y"], 2*np.cumsum(inputs["x"]))


def test_nonzero_start_and_held_slow_module_before_first_update():
    graph = ModuleGraph((accumulator("a", every=3),), inputs=(port("x"),),
                        bindings=(InputBinding("x", ("a", "u")),))
    state = ModuleGraphState(jnp.asarray(1), {"a": value(5.)})
    p = {"a": {"gain": jnp.asarray(1.)}}
    program = CompiledModuleGraph(graph, ticks=4, start_tick=1, physics_dt_s=0.1)
    final, trace = program.run(state, {"x": jnp.array([1., 2., 3., 4.])}, p)
    np.testing.assert_array_equal(trace.outputs["a"]["y"], [5., 5., 8., 8.])
    assert int(final.tick) == 5
    no_events = CompiledModuleGraph(graph, ticks=1, start_tick=1, physics_dt_s=0.1)
    final, _ = no_events.run(state, {"x": jnp.ones(1)}, p)
    assert float(final.modules["a"].state) == 5.


def test_fused_and_unfused_schedules_agree():
    graph = ModuleGraph((accumulator("a"), accumulator("b")),
                        (Wire(("b", "y"), ("a", "u"), 1), Wire(("a", "y"), ("b", "u"))))
    initial = ModuleGraphState(jnp.asarray(0), {"a": value(1.), "b": value(1.)})
    p = {name: {"gain": jnp.asarray(0.2)} for name in graph.modules}
    a = CompiledModuleGraph(graph, ticks=9, physics_dt_s=0.1, chunk_ticks=4).run(initial, {}, p)
    b = CompiledModuleGraph(graph, ticks=9, physics_dt_s=0.1, fuse=False).run(initial, {}, p, max_workers=1)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


def test_graph_assembly_runs_in_world_and_native_scan_holds_signals():
    from aerodrome.composition import GraphAssembly, EntitySpec, WorldSpec, build_world
    graph = ModuleGraph((accumulator("a", every=3),), inputs=(port("x"),),
                        bindings=(InputBinding("x", ("a", "u")),))
    world = build_world(WorldSpec((EntitySpec("entity", GraphAssembly(graph)),), ticks_per_step=2))
    state = world.reset(42, {"entity": {"a": value()}})
    p = {"a": {"gain": jnp.asarray(1.)}}
    params = world.parameters({"entity": p})
    inputs = world.pack({"entity": {"x": jnp.asarray(2.)}})
    world.validate(state, inputs, params)
    final, trace = jax.jit(lambda s: world.rollout(s, inputs, params, steps=4))(state)
    program = CompiledModuleGraph(graph, ticks=8, physics_dt_s=0.01)
    expected, record = program.run(state.entities[0], {"x": jnp.full(8, 2.)}, p)
    native, native_record = program.native(state.entities[0], {"x": jnp.full(8, 2.)}, p)
    np.testing.assert_array_equal(trace.entities[0].outputs["a"]["y"].reshape(-1), record.outputs["a"]["y"])
    np.testing.assert_array_equal(native_record.outputs["a"]["y"], record.outputs["a"]["y"])
    assert float(final.entities[0].modules["a"].state) == float(expected.modules["a"].state) == 6.


def test_closed_loop_host_boundary_matches_full_compiled_scan():
    from compiled_modules import build_case
    graph, state, inputs, params, _ = build_case(ticks=12)
    pure = CompiledModuleGraph(graph, ticks=12, physics_dt_s=0.02)
    reference = pure.native(state, inputs, params)
    graph, state, inputs, params, calls = build_case("host", ticks=12)
    hybrid = CompiledModuleGraph(graph, ticks=12, physics_dt_s=0.02)
    result = hybrid.run(state, inputs, params)
    assert calls == [0, 4, 8]
    for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(reference), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
