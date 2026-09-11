"""Explicit frame maps. R_ab maps vectors in frame b into frame a.

Single-vector kernels: use vmap for batches. Positions include translation;
free vectors do not. All rotations must be proper orthonormal matrices.
"""
from typing import NamedTuple, Any
import jax.numpy as jnp


class Transform(NamedTuple):
    rotation: Any  # R_ab
    translation: Any  # origin of b expressed in a


def transform_vector(rotation, vector):
    return rotation @ vector


def transform_point(transform, point):
    return transform.rotation @ point + transform.translation


def inverse_transform(transform):
    R = transform.rotation.T
    return Transform(R, -R @ transform.translation)


def compose_transforms(a_from_b, b_from_c):
    """Return a_from_c; rightmost transformation is applied first."""
    return Transform(a_from_b.rotation @ b_from_c.rotation,
                     transform_point(a_from_b, b_from_c.translation))


def transform_covariance(rotation, covariance):
    """Rotate a 3x3 vector covariance, not a full position/attitude covariance."""
    return rotation @ covariance @ rotation.T


def transform_wrench(transform, force, moment):
    """Force/moment about origin b -> force/moment about origin a, in a axes."""
    f = transform.rotation @ force
    return f, transform.rotation @ moment + jnp.cross(transform.translation, f)


def point_velocity(velocity_origin, omega, offset):
    """Rigid point velocity; all inputs in the same frame, offset from origin."""
    return velocity_origin + jnp.cross(omega, offset)


def point_acceleration(acceleration_origin, omega, angular_acceleration, offset):
    """Fixed body point: tangential + centripetal terms, no moving-offset terms."""
    return (acceleration_origin + jnp.cross(angular_acceleration, offset)
            + jnp.cross(omega, jnp.cross(omega, offset)))


def ned_to_enu(vector):
    """Same-origin vector or relative position: [N,E,D] -> [E,N,U]."""
    return jnp.stack((vector[1], vector[0], -vector[2]))


enu_to_ned = ned_to_enu


def frd_to_flu(vector):
    """Same-origin body coordinates: forward/right/down -> forward/left/up."""
    return jnp.stack((vector[0], -vector[1], -vector[2]))


flu_to_frd = frd_to_flu


def ecef_to_ned_matrix(latitude_rad, longitude_rad):
    """Local NED basis at geodetic latitude/longitude; ECEF -> NED vectors."""
    s, c = jnp.sin(latitude_rad), jnp.cos(latitude_rad)
    sl, cl = jnp.sin(longitude_rad), jnp.cos(longitude_rad)
    z = jnp.zeros_like(s+sl)
    return jnp.stack((jnp.stack((-s*cl, -s*sl, c)),
                      jnp.stack((-sl, cl, z)), jnp.stack((-c*cl, -c*sl, -s))))


def ecef_to_ned_transform(origin_ecef_m, latitude_rad, longitude_rad):
    """For positions: subtract the local origin before rotating."""
    R = ecef_to_ned_matrix(latitude_rad, longitude_rad)
    return Transform(R, -R @ origin_ecef_m)


def body_to_ned(vector, rotation_nb):
    return rotation_nb @ vector


def ned_to_body(vector, rotation_nb):
    return rotation_nb.T @ vector
