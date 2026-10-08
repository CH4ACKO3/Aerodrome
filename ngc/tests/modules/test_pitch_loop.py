import dataclasses
import numpy as np
import jax
import jax.numpy as jnp
from aerodrome.core.signals import PitchState, PIDState, NavigationSolution, PitchReference
from aerodrome.control.pid import update as control_update
from aerodrome.systems.pitch_loop import initialize, make_step
from aerodrome.runners.native import run_loop, run_scan


def assert_tree_close(left, right):
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_eager_jit_and_timestamps_agree(case):
    schedule, p, initial, goal = case
    step = make_step(schedule)
    eager, eager_trace = run_loop(step, initial, goal, p, 24)
    compiled, trace = jax.jit(lambda s, p: run_scan(step, s, goal, p, 24))(initial, p)
    assert_tree_close(eager_trace, trace)
    assert_tree_close(eager.truth, compiled.truth)
    assert int(compiled.tick) == 24
    np.testing.assert_array_equal(trace.tick, np.arange(24))
    np.testing.assert_allclose(trace.time_s, np.arange(24) * schedule.physics_dt_s)


def test_multirate_hold_and_no_repeated_measurement(case):
    schedule, p, initial, goal = case
    step = make_step(schedule)
    state1, _ = step(initial, goal, p)
    _, record1 = step(state1, goal, p)
    assert not bool(record1.measurement.valid)
    assert_tree_close(record1.navigation.estimate, state1.navigation_prior.mean)
    assert_tree_close(record1.navigation.covariance, state1.navigation_prior.covariance)
    _, trace = run_scan(step, initial, goal, p, 24)
    np.testing.assert_array_equal(trace.measurement.valid, np.arange(24) % 2 == 0)
    np.testing.assert_array_equal(trace.guidance_updated, np.arange(24) % 10 == 0)
    np.testing.assert_array_equal(trace.control_updated, np.arange(24) % 2 == 0)
    np.testing.assert_allclose(trace.command.elevator_rad[1::2], trace.command.elevator_rad[::2])
    np.testing.assert_array_equal(trace.reference.updated_tick, np.arange(24) // 10 * 10)


def test_hidden_pitch_rate_cannot_change_initial_control(case):
    schedule, p, initial, goal = case
    # Equal accessible measurement and belief; different unobserved truth.
    changed = initial._replace(truth=PitchState(initial.truth.pitch_rad, jnp.asarray(4.)))
    step = make_step(schedule)
    _, first = step(initial, goal, p)
    _, second = step(changed, goal, p)
    assert_tree_close(first.navigation, second.navigation)
    assert_tree_close(first.command, second.command)


def test_batched_seeded_rollouts_match_individual_and_reset(case):
    schedule, p, initial, goal = case
    step = make_step(schedule)
    run = jax.jit(lambda s: run_scan(step, s, goal, p, 24))
    states = [initialize(initial.truth, initial.navigation_prior, jax.random.key(seed)) for seed in (1, 2, 3)]
    batched = jax.tree.map(lambda *values: jnp.stack(values), *states)
    _, batch_trace = jax.jit(jax.vmap(run))(batched)
    for index, state in enumerate(states):
        _, single = run(state)
        assert_tree_close(single, jax.tree.map(lambda x: x[index], batch_trace))
    _, first = run(states[0])
    _, repeated = run(states[0])
    assert_tree_close(first, repeated)
    assert not np.array_equal(batch_trace.measurement.pitch_rad[0], batch_trace.measurement.pitch_rad[1])


def test_sensor_noise_does_not_depend_on_control_schedule(case):
    schedule, p, initial, goal = case
    traces = [run_scan(make_step(s), initial, goal, p, 24)[1] for s in (
        schedule, dataclasses.replace(schedule, control_every=4))]
    noises = [np.asarray(t.measurement.pitch_rad - t.truth.pitch_rad)[np.asarray(t.measurement.valid)] for t in traces]
    np.testing.assert_allclose(noises[0], noises[1], atol=1e-15)


def test_covariance_and_command_bounds(case):
    schedule, p, initial, goal = case
    _, trace = run_scan(make_step(schedule), initial, goal, p, 200)
    covariance = np.asarray(trace.navigation.covariance)
    assert np.all(np.isfinite(covariance))
    np.testing.assert_allclose(covariance, covariance.transpose(0, 2, 1), atol=1e-14)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-14
    assert np.max(np.abs(trace.command.elevator_rad)) <= float(p.control.elevator_limit_rad)


def test_pid_does_not_wind_up_against_saturation(case):
    _, p, initial, _ = case
    nav = NavigationSolution(PitchState(jnp.asarray(0.), jnp.asarray(0.)), jnp.eye(2), jnp.asarray(0))
    state, command = control_update(PIDState(jnp.asarray(0.)), nav,
                                    PitchReference(jnp.asarray(10.), jnp.asarray(0)), p.control, 0.1)
    assert float(state.integral_error_rad_s) == 0
    assert float(command.elevator_rad) == float(p.control.elevator_limit_rad)
