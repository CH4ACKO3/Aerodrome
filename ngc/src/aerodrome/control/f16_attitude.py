"""Local F-16 attitude PD, yaw damper and proportional airspeed hold.

Ideal full-state feedback around one straight-flight trim; no integrator,
altitude or heading loop. Gains below demonstrate 150 m/s at 3000 m only.
"""
from typing import NamedTuple

import jax.numpy as jnp

from aerodrome.core.airdata import air_relative_velocity
from aerodrome.models.f16 import BODY, Controls


class Gains(NamedTuple):
    roll_kp: float = .4
    roll_kd_s: float = .15
    pitch_kp: float = .8
    pitch_kd_s: float = .4
    yaw_kd_s: float = .2
    speed_kp_N_per_m_s: float = 1000.


def command(state, trim_state, trim_controls, parameters, gains=Gains()):
    """Compute actual surface angles and thrust; hold between control updates.

    Positive elevator/aileron produce negative pitch/roll moments in this
    airframe, hence positive gains multiply actual-minus-reference errors.
    Body rates approximate Euler angle rates near the level-flight trim.
    """
    phi, theta, _ = BODY.euler_angles(state) - BODY.euler_angles(trim_state)
    p, q, r = state.omega_body_rad_s

    def speed(s):
        return jnp.linalg.norm(air_relative_velocity(s.velocity_body_m_s,
                               parameters.wind_ned_m_s, BODY.rotation_matrix(s)))

    elevator = trim_controls.elevator_rad + gains.pitch_kp*theta + gains.pitch_kd_s*q
    aileron = trim_controls.aileron_rad + gains.roll_kp*phi + gains.roll_kd_s*p
    rudder = trim_controls.rudder_rad + gains.yaw_kd_s*r
    thrust = trim_controls.thrust_N + gains.speed_kp_N_per_m_s*(speed(trim_state)-speed(state))
    return Controls(jnp.clip(elevator, -jnp.deg2rad(25.), jnp.deg2rad(25.)),
                    jnp.clip(aileron, -jnp.deg2rad(21.5), jnp.deg2rad(21.5)),
                    jnp.clip(rudder, -jnp.deg2rad(30.), jnp.deg2rad(30.)),
                    trim_controls.lef_rad, jnp.clip(thrust, 0., 84500.))
