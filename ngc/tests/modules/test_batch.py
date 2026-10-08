from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from aerodrome.composition import EntitySpec, WorldSpec, WorldParameters, build_world, PitchAssembly, PitchInitial
from aerodrome.composition.contracts import SystemSpec
from aerodrome.runners.batch import BatchedWorld
from aerodrome.runners.episodes import EpisodeRunner, Task, Policy, Outcome


class CounterState(NamedTuple):
    value: object
    key: object


@dataclass(frozen=True)
class CounterAssembly:
    backend: str = "jax"
    initial_signals: tuple = ()
    systems: tuple = (SystemSpec("counter", "discrete", state_slots=("value",)),)
    def initialize(self, initial, key):
        return CounterState(initial, key)
    def make_tick(self, schedule):
        def tick(state, action, gain, context):
            following = state._replace(value=state.value + action*gain*context.resources["table"][0])
            return following, state.value
        return tick


def make_counter(*, parameter_axes=None, initial_axes=None, input_axes=0, ticks_per_step=1,
                 initial_sampler=None):
    world = build_world(WorldSpec((EntitySpec("counter", CounterAssembly()),), ticks_per_step=ticks_per_step))
    kwargs = {} if initial_sampler is None else {"initial_sampler": initial_sampler}
    batch = BatchedWorld(world, parameter_axes=parameter_axes, initial_axes=initial_axes,
                         input_axes=input_axes, **kwargs)
    initial = {"counter": jnp.asarray(0.)}
    params = world.parameters({"counter": jnp.asarray(1.)}, resources={"table": jnp.array([1., 999.])})
    return batch, initial, params


def assert_close(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
            x, y = jax.random.key_data(x), jax.random.key_data(y)
        np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-12)


def test_partial_parameter_axes_and_shared_table():
    batch, _, params = make_counter(parameter_axes=WorldParameters((0,), None), initial_axes=0)
    initial = {"counter": jnp.array([1., 2., 3.])}
    state = batch.reset(42, [10, 20, 30], initial)
    params = params._replace(entities=(jnp.array([1., 2., 4.]),))
    actions = (jnp.array([2., 3., 4.]),)
    batch.validate(state, actions, params)
    following, trace = jax.jit(batch.step)(state, actions, params)
    np.testing.assert_allclose(following.world.entities[0].value, [3., 8., 19.])
    assert params.resources["table"].shape == (2,)  # no B copies
    for i in range(3):
        single = jax.tree.map(lambda x: x[i], state.world)
        expected = batch.world.step(single, (actions[0][i],), params._replace(entities=(params.entities[0][i],)))
        assert_close(expected, (jax.tree.map(lambda x: x[i], following.world), jax.tree.map(lambda x: x[i], trace)))


def test_time_major_sequences_and_physics_substeps():
    batch, initial, params = make_counter(ticks_per_step=2)
    state = batch.reset(1, [0, 1, 2], initial)
    actions = (jnp.arange(12., dtype=float).reshape(4, 3),)
    following, trace = jax.jit(lambda s, a, p: batch.rollout(s, a, p, steps=4))(state, actions, params)
    assert trace.tick.shape == (4, 3, 2)
    np.testing.assert_array_equal(following.world.tick, [8, 8, 8])
    np.testing.assert_allclose(following.world.entities[0].value, 2*np.sum(actions[0], axis=0))
    expected = state
    for k in range(4):
        expected, _ = batch.step(expected, (actions[0][k],), params)
    assert_close(following, expected)


def test_shared_inputs_and_compact_or_no_records():
    batch, initial, params = make_counter(input_axes=None)
    state = batch.reset(1, [4, 5, 6], initial)
    following, compact = jax.jit(lambda s: batch.rollout(
        s, (jnp.array([1., 2., 3.]),), params, steps=3,
        record=lambda world, trace: world.entities[0].value))(state)
    assert compact.shape == (3, 3)
    np.testing.assert_allclose(compact[:, 0], [1., 3., 6.])
    no_record = jax.jit(lambda s: batch.rollout_constant(s, (jnp.asarray(2.),), params, steps=3, record=None))(state)
    assert no_record[1] == ()
    assert_close(following, no_record[0])


def test_rng_is_stable_under_reorder_subbatch_and_episode_reset():
    def sample(key, template):
        return {"counter": template["counter"] + jax.random.normal(key)}
    batch, initial, params = make_counter(initial_sampler=sample)
    state = batch.reset(71, [8, 5, 42], initial)
    reordered = batch.reset(71, [42, 8], initial)
    for i, index in enumerate((2, 0)):
        assert_close(jax.tree.map(lambda x: x[i], reordered.world), jax.tree.map(lambda x: x[index], state.world))
    stepped, _ = batch.step(state, (jnp.ones(3),), params)
    reset = jax.jit(batch.reset_where)(stepped, jnp.array([False, True, False]), initial)
    np.testing.assert_array_equal(reset.episode_id, [0, 1, 0])
    np.testing.assert_array_equal(reset.world.tick, [1, 0, 1])
    for i in (0, 2):
        assert_close(jax.tree.map(lambda x: x[i], reset.world), jax.tree.map(lambda x: x[i], stepped.world))
    assert reset.world.entities[0].value[1] != state.world.entities[0].value[1]
    assert_close(batch.reset_where(stepped, jnp.zeros(3, bool), initial), stepped)
    assert_close(batch.reset_where(stepped, jnp.array([False, True, False]), initial), reset)


def test_parameter_jacobian_has_no_cross_world_terms():
    batch, initial, params = make_counter(parameter_axes=WorldParameters((0,), None))
    state = batch.reset(1, [1, 2], initial)
    def result(gains):
        final, _ = batch.rollout_constant(state, (jnp.array([2., 3.]),),
                                          params._replace(entities=(gains,)), steps=4, record=None)
        return final.world.entities[0].value
    jacobian = jax.jit(jax.jacrev(result))(jnp.ones(2))
    np.testing.assert_allclose(jacobian, [[8., 0.], [0., 12.]])


def test_pitch_batch_matches_individual_with_same_actual_keys(case):
    schedule, p, initial, goal = case
    world = build_world(WorldSpec((EntitySpec("aircraft", PitchAssembly()),), schedule, 2))
    batch = BatchedWorld(world)
    conditions = {"aircraft": PitchInitial(initial.truth, initial.navigation_prior)}
    state = batch.reset(42, [10, 11], conditions)
    inputs = world.pack({"aircraft": goal._replace(pitch_rad=jnp.array([0.1, 0.2]))})
    params = world.parameters({"aircraft": p})
    final, trace = jax.jit(lambda s: batch.rollout_constant(s, inputs, params, steps=6))(state)
    for i in range(2):
        one = jax.tree.map(lambda x: x[i], state.world)
        action = jax.tree.map(lambda x: x[i], inputs)
        expected, records = world.rollout(one, action, params, steps=6)
        assert_close(expected, jax.tree.map(lambda x: x[i], final.world))
        assert_close(records, jax.tree.map(lambda x: x[:, i], trace))


def make_episodes():
    batch, initial, params = make_counter()
    task = Task(lambda state, p: state.entities[0].value,
                lambda obs, action, following, p: Outcome(following, following >= p),
                max_episode_steps=3, parameter_axes=0)
    runner = EpisodeRunner(batch, task)
    task_params = jnp.array([2., 100.])
    state = runner.initialize(batch.reset(42, [10, 20], initial), task_params)
    return runner, state, initial, params, task_params


def test_independent_termination_truncation_and_terminal_observations():
    runner, state, initial, params, task_params = make_episodes()
    final, trace = jax.jit(lambda s: runner.rollout(
        s, (jnp.ones((4, 2)),), params, initial, task_params, steps=4))(state)
    np.testing.assert_array_equal(trace.terminated[:, 0], [False, True, False, True])
    np.testing.assert_array_equal(trace.truncated[:, 1], [False, False, True, False])
    np.testing.assert_array_equal(trace.episode_id, [[0, 0], [0, 0], [1, 0], [1, 1]])
    np.testing.assert_array_equal(trace.next_observation[:, 0], [1., 2., 1., 2.])
    np.testing.assert_array_equal(trace.observation[:, 0], [0., 1., 0., 1.])
    assert float(trace.episode_return[1, 0]) == 3.
    assert float(trace.episode_return[2, 1]) == 6.
    np.testing.assert_array_equal(final.observation, [0., 1.])
    np.testing.assert_array_equal(final.batch.episode_id, [2, 1])
    np.testing.assert_array_equal(final.batch.world.tick, [0, 1])
    assert trace.reward.shape == (4, 2)


def test_recurrent_policy_resets_with_its_own_episode():
    runner, state, initial, params, task_params = make_episodes()
    policy = Policy(lambda obs, p, key: jnp.zeros_like(obs),
                    lambda memory, obs, p, key: (memory+1, ((memory+1)*p,)))
    carry = runner.start_policy(state, jnp.asarray(1.), policy=policy)
    final, trace = jax.jit(lambda c: runner.rollout_policy(
        c, jnp.asarray(1.), params, initial, task_params, policy=policy, steps=4))(carry)
    np.testing.assert_array_equal(trace.action[0][:, 0], [1., 2., 1., 2.])
    np.testing.assert_array_equal(trace.action[0][:, 1], [1., 2., 3., 1.])
    np.testing.assert_array_equal(final.policy, [0., 1.])


def test_stochastic_policy_is_invariant_to_rollout_chunking():
    runner, state, initial, params, task_params = make_episodes()
    policy = Policy(lambda obs, p, key: jax.random.uniform(key),
                    lambda memory, obs, p, key: (memory, (jax.random.uniform(key)+memory,)))
    carry = runner.start_policy(state, (), policy=policy)
    def run(c, steps):
        return runner.rollout_policy(c, (), params, initial, task_params, policy=policy, steps=steps)
    final, trace = jax.jit(lambda c: run(c, 6))(carry)
    middle, a = jax.jit(lambda c: run(c, 2))(carry)
    end, b = jax.jit(lambda c: run(c, 4))(middle)
    assert_close(final, end)
    assert_close(trace, jax.tree.map(lambda x, y: jnp.concatenate((x, y)), a, b))


def test_float32_state_and_rewards_keep_scan_carry_dtypes():
    batch, initial, params = make_counter()
    initial = jax.tree.map(lambda x: x.astype(jnp.float32), initial)
    params = jax.tree.map(lambda x: x.astype(jnp.float32), params)
    task = Task(lambda s, p: s.entities[0].value,
                lambda o, a, n, p: Outcome(-n, jnp.asarray(False)), 2, reward_dtype="float32")
    runner = EpisodeRunner(batch, task)
    state = runner.initialize(batch.reset(1, [1, 2], initial), ())
    final, trace = jax.jit(lambda s: runner.rollout(s, (jnp.ones((5, 2), jnp.float32),),
                                                  params, initial, (), steps=5))(state)
    assert final.observation.dtype == jnp.float32 and trace.reward.dtype == jnp.float32


def test_native_module_graph_can_batch_and_rollout():
    from compiled_modules import build_case
    from aerodrome.composition.graph_assembly import GraphAssembly
    graph, graph_state, _, p, _ = build_case()
    world = build_world(WorldSpec((EntitySpec("plane", GraphAssembly(graph)),), ticks_per_step=2))
    batch = BatchedWorld(world)
    state = batch.reset(1, [7, 8], {"plane": graph_state.modules})
    actions = ({"target": jnp.array([4., 12.])},)
    params = world.parameters({"plane": p})
    final, trace = jax.jit(lambda s: batch.rollout_constant(s, actions, params, steps=8))(state)
    for i in range(2):
        expected = world.rollout(jax.tree.map(lambda x: x[i], state.world),
                                 jax.tree.map(lambda x: x[i], actions), params, steps=8)
        assert_close(expected, (jax.tree.map(lambda x: x[i], final.world),
                                jax.tree.map(lambda x: x[:, i], trace)))


def test_pitch_stateless_policy_sees_belief_not_hidden_truth():
    from batch_rollout import build_batch
    from aerodrome.core.signals import PitchGoal
    batch, state, params, initial = build_batch(size=2)
    task = Task(lambda w, p: w.entities[0].navigation_prior.mean.pitch_rad,
                lambda o, a, n, p: Outcome(-n*n, jnp.asarray(False)), 2)
    runner = EpisodeRunner(batch, task)
    policy = Policy(lambda obs, p, key: (), lambda memory, obs, p, key: (memory, (PitchGoal(obs),)))
    carry = runner.start_policy(runner.initialize(state, ()), (), policy=policy)
    final, trace = jax.jit(lambda c: runner.rollout_policy(
        c, (), params, initial, (), policy=policy, steps=3))(carry)
    assert final.policy == ()
    assert state.world.entities[0].truth.pitch_rad[0] != state.world.entities[0].truth.pitch_rad[1]
    np.testing.assert_array_equal(trace.action[0].pitch_rad[0], [0., 0.])
    np.testing.assert_array_equal(final.environment.batch.episode_id, [1, 1])
