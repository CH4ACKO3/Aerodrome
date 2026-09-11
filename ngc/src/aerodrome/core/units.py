"""Explicit boundary conversions; numerical models remain SI/radians.

No runtime unit registry or string parsing in compiled kernels.
"""
import jax.numpy as jnp

FT_TO_M = .3048
KNOT_TO_M_S = 1852. / 3600.
LBF_TO_N = .45359237 * 9.80665
SLUG_TO_KG = LBF_TO_N / FT_TO_M


def feet_to_m(value):
    return jnp.asarray(value) * FT_TO_M


def m_to_feet(value):
    return jnp.asarray(value) / FT_TO_M


def knots_to_m_s(value):
    return jnp.asarray(value) * KNOT_TO_M_S


def m_s_to_knots(value):
    return jnp.asarray(value) / KNOT_TO_M_S


def degrees_to_rad(value):
    return jnp.deg2rad(value)


def rad_to_degrees(value):
    return jnp.rad2deg(value)


def wrap_pi(angle):
    """[-pi, pi); discontinuous at odd multiples of pi."""
    return (jnp.asarray(angle) + jnp.pi) % (2*jnp.pi) - jnp.pi


def wrap_2pi(angle):
    """[0, 2*pi)."""
    return jnp.asarray(angle) % (2*jnp.pi)


def angle_difference(target, current):
    """Shortest signed target-current difference, in [-pi,pi)."""
    return wrap_pi(jnp.asarray(target)-jnp.asarray(current))
