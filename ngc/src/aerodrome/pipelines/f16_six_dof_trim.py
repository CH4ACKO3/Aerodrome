"""Host-side, wings-level straight-flight trim of all six accelerations."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from aerodrome.core.airdata import velocity_from_airdata
from aerodrome.core.rotations import euler321_to_quaternion, quaternion_to_matrix
from aerodrome.models import f16
from aerodrome.models.rigid_body import RigidBodyState


def trim(tables, parameters, *, speed_m_s=150., height_m=3000., heading_rad=0., lef_rad=0.):
    """Return (state, controls) for level flight relative to a uniform air mass.

    Solve alpha, beta, elevator, aileron, rudder and thrust. Small nonzero beta
    accounts for the source's asymmetric lateral coefficients. theta=alpha,
    roll=0 imply zero air-relative vertical speed. Ground speed includes wind.
    Positive NED-down gravity and float64 are required for host trim accuracy.
    """
    if not jax.config.x64_enabled:
        raise ValueError("F16 trim requires JAX float64")
    gravity = np.asarray(parameters.mass.gravity_ned_m_s2)
    if not np.allclose(gravity[:2], 0.) or gravity[2] <= 0:
        raise ValueError("level-flight trim requires positive NED-down gravity")

    def unpack(z):
        alpha, beta, elevator, aileron, rudder, thrust = z
        attitude = euler321_to_quaternion(jnp.array([0., alpha, heading_rad]))
        velocity = velocity_from_airdata(speed_m_s, alpha, beta)
        velocity = velocity + quaternion_to_matrix(attitude).T @ parameters.wind_ned_m_s
        state = RigidBodyState(jnp.array([0., 0., -height_m]), velocity, attitude, jnp.zeros(3))
        return state, f16.Controls(elevator, aileron, rudder, lef_rad, thrust*10000.)

    def residual(z):
        state, controls = unpack(z)
        derivative = f16.rhs(state, controls, tables, parameters)
        return jnp.concatenate((derivative.velocity_body_m_s, 10.*derivative.omega_body_rad_s))

    evaluate = jax.jit(residual)
    jacobian = jax.jit(jax.jacfwd(residual))
    lower = np.r_[np.deg2rad([-5., -10., -25., -21.5, -30.]), 0.]
    upper = np.r_[np.deg2rad([15., 10., 25., 21.5, 30.]), 8.45]
    result = least_squares(lambda z: np.asarray(evaluate(z)), [.05, 0., -.03, 0., 0., 1.],
                           jac=lambda z: np.asarray(jacobian(z)), bounds=(lower, upper),
                           xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
    error = np.asarray(evaluate(result.x))
    if not result.success or np.max(np.abs(error)) > 1e-8:
        raise RuntimeError(f"F16 six-DoF trim failed: {result.message}; residual={error}")
    return unpack(result.x)
