"""F-16 airframe: full table aerodynamics + quaternion 6DoF, SI / FRD / NED.

Coefficients ported from ISRL F16AeroFM.m (MIT, Raktim Bhattacharya, 2023).
See data/f16/LICENSE and docs/f16-six-dof.md for source and modeling choices.
Controls are actual surface angles and external thrust, not actuator commands.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from aerodrome.core.airdata import air_relative_velocity, airdata, dynamic_pressure
from aerodrome.core.lookup import RegularGrid, lookup
from aerodrome.models.f16_longitudinal import density
from aerodrome.models.rigid_body import BodyLoads, RigidBody6DoF, mass_properties


class Geometry(NamedTuple):
    area_m2: Any = 300 * .3048**2
    span_m: Any = 30 * .3048
    chord_m: Any = 11.32 * .3048
    xcg: Any = .30
    xcgr: Any = .35


class Controls(NamedTuple):
    elevator_rad: Any = 0.
    aileron_rad: Any = 0.
    rudder_rad: Any = 0.
    lef_rad: Any = 0.
    thrust_N: Any = 0.


class Parameters(NamedTuple):
    mass: Any
    geometry: Geometry
    wind_ned_m_s: Any


BODY = RigidBody6DoF()


def default_parameters(*, gravity_ned_m_s2=(0., 0., 9.806), wind_ned_m_s=(0., 0., 0.)):
    # Jxz is a product of inertia: the tensor has -Jxz off-diagonal entries.
    inertia = np.array([[9496., 0., -982.], [0., 55814., 0.], [-982., 0., 63100.]])
    mass = mass_properties(636.94 * 14.5939, inertia * 14.5939 * .3048**2, gravity_ned_m_s2)
    return Parameters(mass, Geometry(), jnp.asarray(wind_ned_m_s))


def load_tables(directory=None):
    folder = Path(directory) if directory else Path(__file__).parent / "data/f16"
    manifest = json.loads((folder / "aerodynamics.json").read_text())
    path = folder / "aerodynamics.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["npz_sha256"]:
        raise ValueError("F16 data integrity check failed")
    with np.load(path, allow_pickle=False) as data:
        return {name: RegularGrid(tuple(jnp.asarray(data[axis]) for axis in group["axes"]),
                                  jnp.asarray(data[name])) for name, group in manifest["groups"].items()}


def coefficients(speed_m_s, alpha_rad, beta_rad, omega_body_rad_s, controls, tables, geometry=Geometry()):
    """Return [Cx,Cy,Cz,Cl,Cm,Cn]; body rates in rad/s, table angles in degrees.

    Linear interpolation without extrapolation. The common LEF table envelope
    is alpha [-20,45], beta [-30,30], elevator [-25,25] degrees.
    """
    alpha, beta, elevator = jnp.rad2deg(jnp.array([alpha_rad, beta_rad, controls.elevator_rad]))
    da = jnp.rad2deg(controls.aileron_rad) / 21.5
    dr = jnp.rad2deg(controls.rudder_rad) / 30.
    dl = 1. - jnp.rad2deg(controls.lef_rad) / 25.
    p, q, r = omega_body_rad_s
    b, c = geometry.span_m, geometry.chord_m
    cg = geometry.xcgr - geometry.xcg

    cx, cz, cm = lookup(tables["longitudinal"], (alpha, beta, elevator))
    cx0, cz0, cm0 = lookup(tables["longitudinal"], (alpha, beta, 0.))
    cl, cn = lookup(tables["lateral"], (alpha, beta, elevator))
    cl0, cn0 = lookup(tables["lateral"], (alpha, beta, 0.))
    cy, cy_a, cl_a, cn_a, cy_r, cl_r, cn_r = lookup(tables["side_controls"], (alpha, beta))
    cx_l, cy_l, cz_l, cl_l, cm_l, cn_l, cy_al, cl_al, cn_al = lookup(tables["lef"], (alpha, beta))
    cxq, czq, cmq, cyp, cyr, clp, clr, cnp, cnr, clbeta, cnbeta, dcm = lookup(tables["rates"], (alpha,))
    dcxq, dczq, dcmq, dcyp, dcyr, dclp, dclr, dcnp, dcnr = lookup(tables["lef_rates"], (alpha,))
    eta = lookup(tables["elevator"], (elevator,))[0]

    Cx = cx + (cx_l - cx0)*dl + c/(2*speed_m_s)*(cxq + dcxq*dl)*q
    # Use pitch-rate LEF derivative, not the static Cz LEF increment (upstream typo).
    Cz = cz + (cz_l - cz0)*dl + c/(2*speed_m_s)*(czq + dczq*dl)*q
    Cm = cm*eta + Cz*cg + (cm_l - cm0)*dl + c/(2*speed_m_s)*(cmq + dcmq*dl)*q + dcm

    def lateral(base, neutral, lef, aileron, aileron_lef, rudder, rate_p, rate_r, delta_p, delta_r):
        delta_a = aileron - neutral
        return (base + (lef - neutral)*dl + (delta_a + (aileron_lef - lef - delta_a)*dl)*da
                + (rudder - neutral)*dr
                + b/(2*speed_m_s)*((rate_p + delta_p*dl)*p + (rate_r + delta_r*dl)*r))

    Cy = lateral(cy, cy, cy_l, cy_a, cy_al, cy_r, cyp, cyr, dcyp, dcyr)
    Cl = lateral(cl, cl0, cl_l, cl_a, cl_al, cl_r, clp, clr, dclp, dclr) + clbeta*beta
    Cn = lateral(cn, cn0, cn_l, cn_a, cn_al, cn_r, cnp, cnr, dcnp, dcnr) + cnbeta*beta - Cy*cg*c/b
    return jnp.stack((Cx, Cy, Cz, Cl, Cm, Cn))


def loads(state, controls, tables, parameters):
    """Aerodynamic and thrust loads about COM; gravity is added by the body."""
    relative = air_relative_velocity(state.velocity_body_m_s, parameters.wind_ned_m_s, BODY.rotation_matrix(state))
    air = airdata(relative)
    g = parameters.geometry
    coeff = coefficients(air.speed_m_s, air.alpha_rad, air.beta_rad, state.omega_body_rad_s, controls, tables, g)
    scale = dynamic_pressure(density(-state.position_ned_m[2]), air.speed_m_s) * g.area_m2
    return BodyLoads(scale*coeff[:3] + jnp.array([controls.thrust_N, 0., 0.]),
                     scale*coeff[3:] * jnp.array([g.span_m, g.chord_m, g.span_m]))


def rhs(state, controls, tables, parameters):
    return BODY.rhs(state, loads(state, controls, tables, parameters), parameters.mass)


def step(state, controls, tables, parameters, dt_s):
    """RK4, recomputing atmosphere and aerodynamic loads at every stage."""
    return BODY.step(state, controls, parameters.mass, dt_s,
                     load_fn=lambda t, s, u, mass, resources: loads(s, u, tables, parameters))
