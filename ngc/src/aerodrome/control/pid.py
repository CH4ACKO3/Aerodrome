"""control.pitch_pid: u=kp*(theta_ref-theta_hat)+ki*I-kd*q_hat."""
from typing import NamedTuple
import jax.numpy as jnp
from aerodrome.core.signals import Array, PIDState, ElevatorCommand


class PIDParameters(NamedTuple):
    kp: Array
    ki: Array
    kd: Array
    elevator_limit_rad: Array


def update(state, navigation, reference, params, dt_s):
    """Derivative on estimated pitch; conditional integration anti-windup.

    kp is dimensionless, ki [s^-1], kd [s]. Requires nonnegative gains in
    this demonstrator. Output is held until the next control update.
    """
    error = reference.pitch_rad - navigation.estimate.pitch_rad
    candidate = state.integral_error_rad_s + dt_s * error
    base = params.kp * error - params.kd * navigation.estimate.pitch_rate_rad_s
    raw = base + params.ki * candidate
    drives_outward = ((raw > params.elevator_limit_rad) & (error > 0)) | (
        (raw < -params.elevator_limit_rad) & (error < 0))
    integral = jnp.where(drives_outward | (params.ki == 0), state.integral_error_rad_s, candidate)
    command = jnp.clip(base + params.ki * integral,
                       -params.elevator_limit_rad, params.elevator_limit_rad)
    return PIDState(integral), ElevatorCommand(command)
