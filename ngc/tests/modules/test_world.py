from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aerodrome.composition import EntitySpec, WorldSpec, PitchAssembly, PitchInitial, build_world
from aerodrome.composition.contracts import SystemSpec
from aerodrome.systems.pitch_loop import Blocks, make_step
from aerodrome.runners.native import run_scan
from aerodrome.core.integrators import rk4


def setup(case, ids=("aircraft",)):
    schedule, p, initial, goal = case
    world = build_world(WorldSpec(tuple(EntitySpec(i, PitchAssembly()) for i in ids), schedule, 4))
    state = world.reset(42, {i: PitchInitial(initial.truth, initial.navigation_prior) for i in ids})
    return world, state, world.pack({i: goal for i in ids}), world.parameters({i: p for i in ids})


def close(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-12)


def test_world_matches_original_tick_loop(case):
    world, state, inputs, params = setup(case)
    world.validate(state, inputs, params)
    expected, reference = run_scan(make_step(case[0]), state.entities[0], inputs[0], params.entities[0], 24)
    final, trace = jax.jit(lambda s, p: world.rollout(s, inputs, p, steps=6))(state, params)
    close(final.entities[0], expected)
    close(jax.tree.map(lambda x: x.reshape((24,) + x.shape[2:]), trace.entities[0]), reference)
    assert trace.tick.shape == (6, 4)
    np.testing.assert_array_equal(trace.tick.reshape(-1), np.arange(24))
    assert int(final.tick) == 24 and int(state.tick) == 0


def test_step_equals_ticks_and_holds_goal(case):
    world, state, inputs, params = setup(case)
    expected = state
    for _ in range(4):
        expected, _ = world.tick(expected, inputs, params)
    actual, trace = world.step(state, inputs, params)
    close(expected, actual)
    np.testing.assert_array_equal(trace.entities[0].control_updated, [True, False, True, False])


def test_entity_random_streams_survive_reordering_and_insertion(case):
    one, s1, u1, p1 = setup(case)
    many, s2, u2, p2 = setup(case, ("target", "aircraft"))
    close(s1.entities[0], s2.entities[1])
    _, t1 = one.step(s1, u1, p1)
    _, t2 = many.step(s2, u2, p2)
    close(t1.entities[0], t2.entities[1])
    assert not np.array_equal(t2.entities[0].measurement.pitch_rad, t2.entities[1].measurement.pitch_rad)


def test_vmap_independent_worlds_and_reset(case):
    world, state, inputs, params = setup(case)
    conditions = {"aircraft": PitchInitial(case[2].truth, case[2].navigation_prior)}
    states = jax.vmap(lambda seed: world.reset(seed, conditions))(jnp.arange(3))
    batched = jax.jit(jax.vmap(lambda s: world.step(s, inputs, params)))(states)
    for i in range(3):
        single = world.step(world.reset(i, conditions), inputs, params)
        close(jax.tree.map(lambda x: x[i], batched), single)
    close(world.reset(42, conditions), state)


def test_parameter_gradient_matches_finite_difference(case):
    world, state, inputs, params = setup(case)

    def objective(damping):
        p = params.entities[0]
        altered = p._replace(true_model=p.true_model._replace(damping_s=damping))
        final, _ = world.rollout(state, inputs, params._replace(entities=(altered,)), steps=5)
        return final.entities[0].truth.pitch_rad

    x, epsilon = jnp.asarray(0.8), 1e-5
    gradient = jax.jit(jax.grad(objective))(x)
    finite = (objective(x + epsilon) - objective(x - epsilon)) / (2 * epsilon)
    assert abs(float(gradient)) > 1e-6
    np.testing.assert_allclose(gradient, finite, rtol=1e-5, atol=1e-9)


def test_replace_physics_without_changing_world_or_evaluation(case):
    world, state, inputs, params = setup(case)
    blocks = replace(Blocks(), plant_advance=lambda x, u, p, dt: x)
    frozen = build_world(replace(world.spec, entities=(EntitySpec("aircraft", PitchAssembly(blocks)),)))
    final, _ = frozen.step(state, inputs, params)
    close(final.entities[0].truth, state.entities[0].truth)
    assert float(world.step(state, inputs, params)[0].entities[0].truth.pitch_rad) != float(state.entities[0].truth.pitch_rad)


def test_coupled_pytree_rk4_stage_evaluation():
    # Harmonic oscillator: both derivatives depend on the evolving joint state.
    rhs = lambda t, x, u, p: {"position": x["velocity"], "velocity": -x["position"]}
    initial = {"position": jnp.asarray(1.), "velocity": jnp.asarray(0.)}
    def solve(dt, count):
        return jax.lax.scan(lambda x, _: (rk4(rhs, 0., x, (), (), dt), None), initial, None, count)[0]
    coarse, fine = solve(0.1, 10), solve(0.05, 20)
    error = lambda x: np.linalg.norm([float(x["position"])-np.cos(1), float(x["velocity"])+np.sin(1)])
    assert error(coarse) / error(fine) > 15


def test_heterogeneous_entity_and_numeric_shared_resource(case):
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class ClockAssembly:
        backend: str = "jax"
        initial_signals: tuple = ()
        systems: tuple = (SystemSpec("clock", "discrete", state_slots=("elapsed",)),)
        def initialize(self, initial_conditions, key):
            return initial_conditions
        def make_tick(self, schedule):
            def tick(state, inputs, parameters, context):
                following = state + context.dt_s * context.resources["scale"]
                return following, state
            return tick
    schedule, p, initial, goal = case
    world = build_world(WorldSpec((EntitySpec("aircraft", PitchAssembly()),
                                  EntitySpec("clock", ClockAssembly())), schedule, 2))
    state = world.reset(2, {"aircraft": PitchInitial(initial.truth, initial.navigation_prior),
                           "clock": jnp.asarray(0.)})
    params = world.parameters({"aircraft": p, "clock": ()}, resources={"scale": jnp.asarray(3.)})
    inputs = world.pack({"aircraft": goal, "clock": ()})
    world.validate(state, inputs, params)
    final, _ = jax.jit(world.step)(state, inputs, params)
    np.testing.assert_allclose(world.entity_state(final, "clock"), 0.06)


@pytest.mark.parametrize("chunk_ticks", [1, 5, 64])
def test_graph_world_matches_monolithic_rollout_from_nonzero_time(case, chunk_ticks):
    from aerodrome.runners.graph_world import GraphWorldRunner
    world, state, inputs, parameters = setup(case, ("aircraft", "target"))
    state, _ = world.step(state, inputs, parameters)
    expected = jax.jit(lambda s, p: world.rollout(s, inputs, p, steps=3))(state, parameters)
    runner = GraphWorldRunner(world, max_workers=2, chunk_ticks=chunk_ticks)
    actual = runner.run(state, inputs, parameters, steps=3)
    close(actual, expected)
    assert int(actual[0].tick) == 16
