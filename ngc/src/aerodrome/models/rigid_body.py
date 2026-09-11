"""Fixed-mass 6DoF in a locally inertial NED frame; SI, radians, FRD body axes.

R_nb maps body vectors into NED. Quaternions are Hamilton [w,x,y,z],
body-to-NED; Euler321 states are [roll,pitch,yaw], R_nb=Rz(yaw)Ry(pitch)Rx(roll).
Loads exclude gravity and moments are about the centre of mass.
"""
from dataclasses import dataclass
from typing import Any, NamedTuple
import jax.numpy as jnp
import numpy as np
from aerodrome.core.integrators import rk4
from aerodrome.core.rotations import (
    normalize_quaternion, euler321_to_quaternion, quaternion_to_matrix,
    quaternion_to_euler321, body_rates_to_euler321_rates)


class RigidBodyState(NamedTuple):
    position_ned_m: Any
    velocity_body_m_s: Any
    attitude: Any
    omega_body_rad_s: Any


class MassProperties(NamedTuple):
    mass_kg: Any
    inertia_body_kg_m2: Any
    gravity_ned_m_s2: Any


class BodyLoads(NamedTuple):
    force_body_N: Any
    moment_body_Nm: Any


def mass_properties(mass_kg, inertia_body_kg_m2, gravity_ned_m_s2=(0., 0., 9.80665)):
    """Host validation, once before tracing. Supply the actual inertia matrix.

    Off-diagonal entries are tensor entries (often minus products of inertia).
    Runtime parameters remain differentiable PyTree leaves; callers updating
    them directly must preserve these constraints.
    """
    m, I, g = (np.asarray(x, dtype=float) for x in
               (mass_kg, inertia_body_kg_m2, gravity_ned_m_s2))
    if m.shape != () or not np.isfinite(m) or m <= 0:
        raise ValueError("mass must be a finite positive scalar")
    if I.shape != (3, 3) or not np.all(np.isfinite(I)) or not np.allclose(I, I.T, rtol=1e-12, atol=1e-12):
        raise ValueError("inertia must be a finite symmetric 3x3 tensor")
    moments = np.linalg.eigvalsh(I)
    if moments[0] <= 0 or moments[-1] > moments[0]+moments[1]+1e-12*moments[-1]:
        raise ValueError("inertia must be positive definite with physical principal moments")
    if g.shape != (3,) or not np.all(np.isfinite(g)):
        raise ValueError("gravity must be a finite NED vector")
    return MassProperties(*map(jnp.asarray, (m, I, g)))


def force_at_point(force_body_N, point_from_com_body_m, moment_body_Nm=None):
    """Translate a force at a body-fixed point to an equivalent COM wrench."""
    force = jnp.asarray(force_body_N)
    moment = jnp.zeros_like(force) if moment_body_Nm is None else jnp.asarray(moment_body_Nm)
    return BodyLoads(force, moment+jnp.cross(jnp.asarray(point_from_com_body_m), force))


def sum_loads(*loads):
    """Compose aero/engine/etc. loads; at least one contribution is required."""
    if not loads:
        raise ValueError("supply at least one load")
    return BodyLoads(sum(x.force_body_N for x in loads), sum(x.moment_body_Nm for x in loads))


def held_loads(time_s, state, inputs, parameters, resources):
    """Default load law: external BodyLoads held for the whole integration step."""
    return inputs


@dataclass(frozen=True)
class RigidBody6DoF:
    attitude: str = "quaternion"
    euler_singularity_cos: float = 1e-6

    def __post_init__(self):
        if self.attitude not in ("quaternion", "euler321"):
            raise ValueError("attitude must be quaternion or euler321")
        if not 0 < self.euler_singularity_cos < 1:
            raise ValueError("Euler singularity threshold must be in (0,1)")

    def initialize(self, *, position_ned_m=(0., 0., 0.), velocity_body_m_s=(0., 0., 0.),
                   euler_rad=None, quaternion=None, omega_body_rad_s=(0., 0., 0.)):
        """Host initializer; accepts one orientation convention regardless of storage."""
        if euler_rad is not None and quaternion is not None:
            raise ValueError("provide Euler angles or quaternion, not both")
        vectors = [np.asarray(v, dtype=float) for v in (position_ned_m, velocity_body_m_s, omega_body_rad_s)]
        if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in vectors):
            raise ValueError("position, velocity and omega must be finite 3-vectors")
        if quaternion is not None:
            q = np.asarray(quaternion, dtype=float)
            if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) <= 0:
                raise ValueError("quaternion must be a finite nonzero 4-vector")
            q = normalize_quaternion(jnp.asarray(q))
            e = quaternion_to_euler321(q)
        else:
            e = np.asarray((0., 0., 0.) if euler_rad is None else euler_rad, dtype=float)
            if e.shape != (3,) or not np.all(np.isfinite(e)):
                raise ValueError("Euler angles must be a finite 3-vector in radians")
            e = jnp.asarray(e)
            q = euler321_to_quaternion(e)
        if self.attitude == "euler321" and abs(float(jnp.cos(e[1]))) <= self.euler_singularity_cos:
            raise ValueError("Euler attitude is at gimbal lock; use quaternion storage")
        pos, vel, omega = map(jnp.asarray, vectors)
        return RigidBodyState(pos, vel, q if self.attitude == "quaternion" else e, omega)

    def rotation_matrix(self, state):
        return quaternion_to_matrix(state.attitude if self.attitude == "quaternion"
                                    else euler321_to_quaternion(state.attitude))

    def euler_angles(self, state):
        return quaternion_to_euler321(state.attitude) if self.attitude == "quaternion" else state.attitude

    def rhs(self, state, loads, parameters):
        """Newton-Euler derivative. No aerodynamic or propulsion assumptions."""
        expected = 4 if self.attitude == "quaternion" else 3
        if state.attitude.shape != (expected,) or any(x.shape != (3,) for x in
                (state.position_ned_m, state.velocity_body_m_s, state.omega_body_rad_s,
                 loads.force_body_N, loads.moment_body_Nm)):
            raise ValueError("rigid-body state/load shape mismatch; batch with vmap")
        R = self.rotation_matrix(state)
        v, omega = state.velocity_body_m_s, state.omega_body_rad_s
        I = parameters.inertia_body_kg_m2
        dv = loads.force_body_N/parameters.mass_kg + R.T@parameters.gravity_ned_m_s2-jnp.cross(omega, v)
        dw = jnp.linalg.solve(I, loads.moment_body_Nm-jnp.cross(omega, I@omega))
        if self.attitude == "quaternion":
            w, xyz = state.attitude[0], state.attitude[1:]
            da = .5*jnp.concatenate((-jnp.dot(xyz, omega)[None], w*omega+jnp.cross(xyz, omega)))
        else:
            da = body_rates_to_euler321_rates(state.attitude, omega, self.euler_singularity_cos)
        return RigidBodyState(R@v, dv, da, dw)

    def step(self, state, inputs, parameters, dt_s, *, time_s=0., load_fn=held_loads, resources=()):
        """RK4 with stage-wise loads. load_fn is a static pure JAX callable.

        load_fn(t, stage_state, held_inputs, mass_properties, resources) -> BodyLoads.
        Use resources for explicit numeric aerodynamic/engine parameters.
        """
        def derivative(t, s, u, p):
            return self.rhs(s, load_fn(t, s, u, p, resources), p)
        result = rk4(derivative, time_s, state, inputs, parameters, dt_s)
        if self.attitude == "quaternion":
            result = result._replace(attitude=normalize_quaternion(result.attitude))
        return result
