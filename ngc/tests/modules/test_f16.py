"""Whole-airframe numerical behavior: source data, reduction, trim and motion."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from aerodrome.core.airdata import airdata, velocity_from_airdata
from aerodrome.models import f16, f16_longitudinal
from aerodrome.pipelines.f16_six_dof_trim import trim


def test_full_aerodynamics_against_archived_julia_motion():
    reference = json.loads((Path(__file__).parents[1]/"data/f16_julia_initial.json").read_text())
    states = np.asarray(reference["state"])
    dt = reference["time_s"][1] - reference["time_s"][0]
    # Fifth-order forward derivative of actual upstream solver output, not a
    # second implementation of the aerodynamic equations. p=q=r=0 initially.
    dx = np.array([-137, 300, -300, 200, -75, 12]) @ states / (60*dt)
    rho = .002377*(1 - .703e-5*10000)**4.14  # original Julia atmosphere, slug/ft^3
    qS = .5*rho*300**2*300
    inertia = np.array([[9496., 0., -982.], [0., 55814., 0.], [-982., 0., 63100.]])
    expected = np.r_[(636.94*dx[6]-9000)/qS, 636.94*300*dx[8]/qS,
                     636.94*(300*dx[7]-32.17)/qS,
                     (inertia @ dx[9:12])/qS/np.array([30., 11.32, 30.])]
    tables, parameters = f16.load_tables(), f16.default_parameters()
    controls = f16.Controls(*[np.deg2rad(2.)]*4, thrust_N=9000*14.5939*.3048)
    actual = f16.coefficients(300*.3048, 0., 0., jnp.zeros(3), controls, tables)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    # Check the coupled inertia sign against the same independent angular motion.
    state = f16.BODY.initialize(position_ned_m=(0., 0., -3048.), velocity_body_m_s=(91.44, 0., 0.))
    derivative = f16.rhs(state, controls, tables, parameters)
    density_ratio = f16_longitudinal.density(3048.)/(rho*14.5939/.3048**3)
    np.testing.assert_allclose(derivative.omega_body_rad_s, dx[9:12]*density_ratio, rtol=0, atol=2e-6)


def test_aerodynamic_tables_and_longitudinal_reduction():
    tables = f16.load_tables()
    from aerodrome.core.lookup import lookup
    rng = np.random.default_rng(24)
    # All 43 channels at endpoints and off-grid points, independently interpolated.
    for grid in tables.values():
        axes = tuple(map(np.asarray, grid.axes))
        points = np.stack([rng.uniform(axis[0], axis[-1], 8) for axis in axes], axis=-1)
        points = np.vstack(([axis[0] for axis in axes], points, [axis[-1] for axis in axes]))
        expected = RegularGridInterpolator(axes, np.asarray(grid.values))(points)
        np.testing.assert_allclose(jax.jit(jax.vmap(lambda x: lookup(grid, x)))(points), expected, atol=1e-12)
    longitudinal = f16_longitudinal.load_tables()
    for alpha, elevator, q, lef in ((.09, -.04, .1, 0.), (.2, .07, -.15, .2), (-.04, .03, .05, .4)):
        controls = f16.Controls(elevator_rad=elevator, lef_rad=lef)
        actual = f16.coefficients(150., alpha, 0., jnp.array([0., q, 0.]), controls, tables)
        expected = f16_longitudinal.coefficients(150., alpha, q, elevator, longitudinal,
                                                f16_longitudinal.Airframe(lef_rad=lef))
        np.testing.assert_allclose(actual[jnp.array([0, 2, 4])], expected, rtol=1e-12, atol=1e-12)


def test_trim_motion_wind_and_batched_differentiation():
    tables, parameters = f16.load_tables(), f16.default_parameters()
    state, controls = trim(tables, parameters)
    derivative = f16.rhs(state, controls, tables, parameters)
    np.testing.assert_allclose(derivative.velocity_body_m_s, 0., atol=1e-8)
    np.testing.assert_allclose(derivative.omega_body_rad_s, 0., atol=1e-9)
    np.testing.assert_allclose(derivative.position_ned_m[2], 0., atol=1e-10)
    # Galilean invariance: change uniform wind and ground velocity together.
    wind = jnp.array([12., -5., 0.])
    windy = parameters._replace(wind_ned_m_s=wind)
    translated = state._replace(velocity_body_m_s=state.velocity_body_m_s+f16.BODY.rotation_matrix(state).T@wind)
    wind_rhs = f16.rhs(translated, controls, tables, windy)
    np.testing.assert_allclose(wind_rhs.velocity_body_m_s, derivative.velocity_body_m_s, atol=1e-12)
    np.testing.assert_allclose(wind_rhs.position_ned_m, derivative.position_ned_m+wind, atol=1e-12)

    def acceleration(offset):
        applied = controls._replace(aileron_rad=controls.aileron_rad+offset)
        return f16.rhs(state, applied, tables, parameters).omega_body_rad_s

    offsets = jnp.array([-.02, 0., .02])
    response = jax.jit(jax.vmap(acceleration))(offsets)
    assert response[0, 0] > 0. and response[2, 0] < 0.  # source aileron sign
    np.testing.assert_allclose(jax.jit(jax.jacfwd(acceleration))(0.),
                               (acceleration(1e-5)-acceleration(-1e-5))/(2e-5), atol=1e-9)


def test_coupled_flight_integrator_convergence():
    tables, parameters = f16.load_tables(), f16.default_parameters()
    # Smooth cell interior: isolate RK4 convergence from table-slope discontinuities.
    state = f16.BODY.initialize(position_ned_m=(0., 0., -3000.),
                velocity_body_m_s=velocity_from_airdata(150., .06, .015),
                euler_rad=(.02, .07, .1), omega_body_rad_s=(.02, -.015, .01))
    controls = f16.Controls(-.025, .005, -.008, .1, 8500.)
    def integrate(dt, count):
        def advance(s, _):
            return f16.step(s, controls, tables, parameters, dt), None
        return jax.jit(lambda: jax.lax.scan(advance, state, None, length=count)[0])()
    solutions = [integrate(.08/n, n) for n in (1, 2, 4, 8)]
    def difference(a, b):
        return np.linalg.norm(np.concatenate([np.ravel(x-y) for x, y in zip(a, b, strict=True)]))
    errors = [difference(s, solutions[-1]) for s in solutions[:-1]]
    assert errors[0]/errors[1] > 12. and errors[1]/errors[2] > 12.
    np.testing.assert_allclose(np.linalg.norm(solutions[-1].attitude), 1., atol=1e-14)
