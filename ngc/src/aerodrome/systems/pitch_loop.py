"""Pitch loop composition. See docs/architecture.md for tick semantics."""
from dataclasses import dataclass
from typing import Callable, NamedTuple
import jax
import jax.numpy as jnp
from aerodrome.core.signals import (
    Array, PitchState, GaussianState, PIDState, PitchReference,
    ElevatorCommand, NavigationSolution, TickRecord,
)
from aerodrome.models import pitch
from aerodrome.models.sensors import measure_pitch
from aerodrome.navigation import kalman
from aerodrome.guidance import pitch as pitch_guidance
from aerodrome.control import pid


class Parameters(NamedTuple):
    true_model: pitch.PitchParameters
    navigation: kalman.KalmanParameters
    control: pid.PIDParameters
    sensor_std_rad: Array
    reference_rate_rad_s: Array


class LoopState(NamedTuple):
    tick: Array
    truth: PitchState
    navigation_prior: GaussianState  # prior at this tick, before measurement
    control: PIDState
    reference: PitchReference
    command: ElevatorCommand
    sensor_key: Array


@dataclass(frozen=True)
class Blocks:
    """Static function wiring. Replacements must preserve this loop's types.

    A different state dimension requires a new explicit system composition.
    """
    navigation_predict: Callable = kalman.predict
    navigation_correct: Callable = kalman.correct
    guidance_update: Callable = pitch_guidance.update
    control_update: Callable = pid.update
    sensor_measure: Callable = measure_pitch
    actuator_evaluate: Callable = pitch.ideal_servo
    plant_advance: Callable = pitch.advance


def initialize(truth, navigation_prior, key):
    """Separate explicit prior: never initialize an estimator from hidden truth."""
    zero = jnp.zeros_like(truth.pitch_rad)
    return LoopState(jnp.asarray(0, dtype=jnp.int32), truth, navigation_prior,
                     PIDState(zero), PitchReference(zero, jnp.asarray(-1, jnp.int32)),
                     ElevatorCommand(zero), key)


def make_step(schedule, blocks=Blocks()):
    """Capture static schedule/functions; numerical parameters remain arguments.

    At t_k: sample -> correct navigation -> guidance -> control -> log.
    Then advance truth and navigation prior to t_(k+1) under the held input.
    """
    def step(state, goal, params):
        k = state.tick
        measurement = blocks.sensor_measure(state.truth, state.sensor_key, k,
                                    params.sensor_std_rad, k % schedule.sensor_every == 0)
        posterior = blocks.navigation_correct(state.navigation_prior, measurement, params.navigation)
        navigation = NavigationSolution(posterior.mean, posterior.covariance, k)
        guidance_due = k % schedule.guidance_every == 0
        reference = jax.lax.cond(
            guidance_due,
            lambda: blocks.guidance_update(state.reference, navigation, goal,
                                            params.reference_rate_rad_s, schedule.guidance_dt_s, k),
            lambda: state.reference,
        )
        control_due = k % schedule.control_every == 0
        control_state, command = jax.lax.cond(
            control_due,
            lambda: blocks.control_update(state.control, navigation, reference,
                                           params.control, schedule.control_dt_s),
            lambda: (state.control, state.command),
        )
        actuator = blocks.actuator_evaluate(command, params.control.elevator_limit_rad)
        record = TickRecord(k, k * schedule.physics_dt_s, state.truth, measurement,
                            navigation, reference, command, actuator, guidance_due, control_due)
        truth = blocks.plant_advance(state.truth, actuator, params.true_model, schedule.physics_dt_s)
        # Here the actuator is ideal and known from the command. A future
        # uncertain servo needs its own nominal model or actuator measurement.
        prior = blocks.navigation_predict(posterior, actuator, params.navigation)
        return LoopState(k+1, truth, prior, control_state, reference, command, state.sensor_key), record
    return step
