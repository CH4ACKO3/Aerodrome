"""Exact constant-acceleration motion for the mathematical teaching projects.

This is a point mass, not a fixed-wing or rotorcraft dynamics model. Inputs are
actual accelerations after any controller limits, disturbances or actuator
efficiency changes. Gravity is not added implicitly.

The same arithmetic works with NumPy arrays in host-side planners and with JAX
arrays under jit/vmap/scan. State and acceleration must use compatible shapes:
for example (2,) for one planar agent or (agents, 2) for a team. Scalar examples
are also supported. There is no backend dispatch or implicit array conversion.
"""
from typing import Any, NamedTuple


class PointMassState(NamedTuple):
    position_m: Any
    velocity_m_s: Any


def step(state, acceleration_m_s2, dt_s):
    """Advance under acceleration held constant for dt_s seconds.

    Integrating v'=a and p'=v gives the exact zero-order-hold update. In
    particular, position uses the OLD velocity plus a*dt²/2, not the new
    velocity: this avoids the extra a*dt²/2 error of semi-implicit Euler.

    Bounds belong to the controller or actuator model. Keeping them outside
    this function lets several projects share one physical update without
    silently changing the commanded experiment.
    """
    return PointMassState(
        state.position_m + state.velocity_m_s*dt_s + .5*acceleration_m_s2*dt_s**2,
        state.velocity_m_s + acceleration_m_s2*dt_s,
    )
