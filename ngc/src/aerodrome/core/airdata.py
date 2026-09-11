"""FRD body and wind-axis kinematics. Air velocity = ground velocity - wind.

Angles are radians. Wind x aligns with aircraft velocity relative to air;
wind z is down. Forces in wind axes are [-drag, side_force, -lift].
"""
from typing import NamedTuple, Any
import jax.numpy as jnp


class AirData(NamedTuple):
    speed_m_s: Any
    alpha_rad: Any
    beta_rad: Any
    angles_valid: Any


def air_relative_velocity(velocity_body, wind_ned, rotation_nb):
    """wind_ned is air-mass velocity, NOT meteorological 'wind from' bearing."""
    return velocity_body - rotation_nb.T @ wind_ned


def airdata(velocity_air_body, min_speed_m_s=1e-6):
    """Angles undefined for near-zero speed or pure lateral flow: NaN + mask."""
    u, v, w = velocity_air_body
    longitudinal = jnp.hypot(u, w)
    speed = jnp.linalg.norm(velocity_air_body)
    valid = (speed > min_speed_m_s) & (longitudinal > min_speed_m_s) & jnp.isfinite(speed)
    alpha = jnp.arctan2(jnp.where(valid, w, 0.), jnp.where(valid, u, 1.))
    beta = jnp.arctan2(jnp.where(valid, v, 0.), jnp.where(valid, longitudinal, 1.))
    return AirData(speed, jnp.where(valid, alpha, jnp.nan), jnp.where(valid, beta, jnp.nan), valid)


def wind_to_body_matrix(alpha_rad, beta_rad):
    ca, sa = jnp.cos(alpha_rad), jnp.sin(alpha_rad)
    cb, sb = jnp.cos(beta_rad), jnp.sin(beta_rad)
    z = jnp.zeros_like(ca+cb)
    return jnp.stack((jnp.stack((ca*cb, -ca*sb, -sa)),
                      jnp.stack((sb, cb, z)), jnp.stack((sa*cb, -sa*sb, ca))))


def body_to_wind_matrix(alpha_rad, beta_rad):
    return wind_to_body_matrix(alpha_rad, beta_rad).T


def velocity_from_airdata(speed_m_s, alpha_rad, beta_rad):
    return speed_m_s * wind_to_body_matrix(alpha_rad, beta_rad)[:, 0]


def aerodynamic_force_body(drag_N, side_force_N, lift_N, alpha_rad, beta_rad):
    return wind_to_body_matrix(alpha_rad, beta_rad) @ jnp.stack((-drag_N, side_force_N, -lift_N))


def dynamic_pressure(density_kg_m3, speed_m_s):
    return .5 * density_kg_m3 * speed_m_s**2


def flight_path_angles(velocity_ned, min_speed_m_s=1e-6):
    """Return (course clockwise from North, climb angle), radians.

    Course is NaN for vertical/stationary motion; climb is NaN at rest.
    """
    n, e, d = velocity_ned
    horizontal = jnp.hypot(n, e)
    course = jnp.arctan2(jnp.where(horizontal > min_speed_m_s, e, 0.),
                        jnp.where(horizontal > min_speed_m_s, n, 1.))
    climb = jnp.arctan2(-d, horizontal)
    return (jnp.where(horizontal > min_speed_m_s, course, jnp.nan),
            jnp.where(jnp.linalg.norm(velocity_ned) > min_speed_m_s, climb, jnp.nan))
