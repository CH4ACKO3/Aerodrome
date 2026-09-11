"""Equation pitch.linear: theta_dot=q; q_dot=-a*theta-b*q+c*delta.

A pedagogical second-order pitch model, NOT an identified aircraft model.
Positive delta is defined to produce positive pitching acceleration.
Parameters a,b,c have units s^-2, s^-1, s^-2 respectively.
"""
from typing import NamedTuple
import jax.numpy as jnp
from aerodrome.core.integrators import rk4
from aerodrome.core.signals import (
    Array, PitchState, ElevatorCommand, ElevatorPosition, to_vector, from_vector,
)


class PitchParameters(NamedTuple):
    stiffness_s2: Array
    damping_s: Array
    elevator_gain_s2: Array


def state_matrix(params):
    a, b, _ = params
    return jnp.array([[0., 1.], [-a, -b]])


def rhs(state: PitchState, actuator: ElevatorPosition, params: PitchParameters):
    """pitch.linear: directly inspectable continuous state derivative."""
    theta, q = state
    a, b, c = params
    return PitchState(q, -a * theta - b * q + c * actuator.elevator_rad)


def advance(state, actuator, params, dt):
    """integration.rk4: reevaluate the RHS at every stage."""
    return rk4(lambda t, x, u, p: rhs(x, u, p), 0., state, actuator, params, dt)


def ideal_servo(command: ElevatorCommand, limit_rad: Array):
    """Known, memoryless saturation; no actuator lag in this example."""
    return ElevatorPosition(jnp.clip(command.elevator_rad, -limit_rad, limit_rad))
