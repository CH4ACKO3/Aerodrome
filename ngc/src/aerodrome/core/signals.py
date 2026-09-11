"""Named JAX PyTrees. Angles are radians; q is pitch rate in rad/s.

These signals describe the pitch example, not a universal six-DOF schema.
Only to_vector/from_vector define the matrix state order [theta, q].
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

Array = jax.Array


class PitchState(NamedTuple):
    pitch_rad: Array
    pitch_rate_rad_s: Array


def to_vector(state: PitchState) -> Array:
    return jnp.stack((state.pitch_rad, state.pitch_rate_rad_s))


def from_vector(value: Array) -> PitchState:
    return PitchState(value[0], value[1])


class PitchMeasurement(NamedTuple):
    pitch_rad: Array
    sample_tick: Array
    valid: Array


class GaussianState(NamedTuple):
    mean: PitchState
    covariance: Array  # axes [theta, q], units follow their outer product


class NavigationSolution(NamedTuple):
    estimate: PitchState
    covariance: Array
    tick: Array


class PitchGoal(NamedTuple):
    pitch_rad: Array


class PitchReference(NamedTuple):
    pitch_rad: Array
    updated_tick: Array


class ElevatorCommand(NamedTuple):
    elevator_rad: Array


class ElevatorPosition(NamedTuple):
    elevator_rad: Array


class PIDState(NamedTuple):
    integral_error_rad_s: Array


class TickRecord(NamedTuple):
    tick: Array  # all signals refer to the beginning of the interval
    time_s: Array
    truth: PitchState
    measurement: PitchMeasurement
    navigation: NavigationSolution
    reference: PitchReference
    command: ElevatorCommand
    actuator: ElevatorPosition
    guidance_updated: Array
    control_updated: Array
