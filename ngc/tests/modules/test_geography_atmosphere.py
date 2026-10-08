import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator
from aerodrome.core import geodesy as geo
from aerodrome.core.lookup import regular_grid,lookup
from aerodrome.models import atmosphere as atm


def test_wgs84_reference_points():
    queries = jnp.array([[0.,0.,0.],[0.,jnp.pi/2,100.],[jnp.pi/2,0.,0.],[jnp.pi/4,0.,0.]])
    expected = [[6378137.,0,0],[0,6378237.,0],[0,0,6356752.314245179],
                [4517590.878848932,0,4487348.408865919]]
    np.testing.assert_allclose(jax.jit(jax.vmap(geo.geodetic_to_ecef))(queries),expected,atol=2e-8)


def test_lla_roundtrip_dateline_poles_and_radius():
    rng = np.random.default_rng(17)
    points = np.column_stack((rng.uniform(-np.pi/2,np.pi/2,100),rng.uniform(-np.pi,np.pi,100),rng.uniform(-900.,1e7,100)))
    points = np.vstack((points,[[np.pi/2,0,100],[-np.pi/2,0,0],[0,np.pi,500]]))
    recovered = jax.jit(jax.vmap(lambda x:geo.ecef_to_geodetic(geo.geodetic_to_ecef(x))))(jnp.asarray(points))
    np.testing.assert_allclose(recovered[:,:2],points[:,:2],atol=1e-13)
    np.testing.assert_allclose(recovered[:,2],points[:,2],atol=1e-8)
    radius = jax.vmap(lambda x:geo.geocentric_radius(x[0],x[2]))(jnp.asarray(points))
    heights = jax.jit(jax.vmap(geo.radius_to_ellipsoid_height))(radius,jnp.asarray(points[:,0]))
    np.testing.assert_allclose(heights,points[:,2],atol=1e-8)
    assert np.all(np.isnan(geo.ecef_to_geodetic(jnp.zeros(3))))
    assert np.isnan(geo.radius_to_ellipsoid_height(0.,0.))


def test_ned_heading_and_geodetic_gradients():
    origin = jnp.array([.4,3.14159,100.])
    offset = jnp.array([100.,200.,-50.])
    target = geo.ned_to_geodetic(offset,origin)
    np.testing.assert_allclose(geo.geodetic_to_ned(target,origin),offset,atol=3e-9)
    relative,az,el,r = geo.relative_position(target,origin,jnp.pi/2)
    np.testing.assert_allclose(relative,[200,-100,-50],atol=3e-9)
    np.testing.assert_allclose([az,el,r],[np.arctan2(-100,200),np.arctan2(50,np.hypot(100,200)),np.linalg.norm(offset)],atol=3e-9)
    jac = jax.jacfwd(lambda x:geo.ecef_to_geodetic(geo.geodetic_to_ecef(x)))(origin)
    np.testing.assert_allclose(jac,np.eye(3),atol=2e-9)


def test_heights_and_lookup_tables():
    height = jnp.array([-500.,0.,11000.,80000.])
    np.testing.assert_allclose(geo.geopotential_to_geometric(geo.geometric_to_geopotential(height)),height,atol=1e-10)
    assert geo.ellipsoid_to_orthometric(103.,23.) == 80.
    assert geo.orthometric_to_ellipsoid(80.,23.) == 103.
    assert geo.height_above_ground(80.,30.) == 50.
    geoid = regular_grid(([-.5,.5],[-1.,1.]),[[10,20],[30,40]])
    assert float(geo.height_from_geoid_grid(jnp.array([0.,0.,100.]),geoid)) == pytest.approx(75.)
    # User-generated radius-to-height table at a specified latitude.
    heights = jnp.linspace(0.,10000.,101)
    radii = jax.vmap(lambda h:geo.geocentric_radius(.6,h))(heights)
    table = regular_grid((radii,),heights)
    assert float(lookup(table,(geo.geocentric_radius(.6,3450.),))) == pytest.approx(3450.,abs=1e-7)


@pytest.mark.parametrize("dimension",[1,2,3,4])
def test_grid_against_scipy_and_boundary(dimension):
    axes = tuple(np.linspace(-1.,1.,i+3) for i in range(dimension))
    values = np.random.default_rng(8).normal(size=tuple(map(len,axes))+(2,))
    grid = regular_grid(axes,values)
    points = np.random.default_rng(9).uniform(-1,1,(10,dimension))
    result = jax.jit(jax.vmap(lambda x:lookup(grid,x)))(jnp.asarray(points))
    np.testing.assert_allclose(result,RegularGridInterpolator(axes,values)(points),atol=1e-14)
    np.testing.assert_allclose(lookup(grid,jnp.ones(dimension)),values[tuple(-1 for _ in axes)])
    assert np.all(np.isnan(lookup(grid,jnp.full(dimension,1.1))))


def test_standard_atmosphere_reference_and_inverse():
    # US1976 geopotential layer bases; pressure rounded reference values.
    H = jnp.array([0.,11000.,20000.,32000.,47000.,51000.,71000.])
    z = geo.geopotential_to_geometric(H)
    T,P = jax.jit(jax.vmap(atm.standard_temperature_pressure))(z)
    np.testing.assert_allclose(T,[288.15,216.65,216.65,228.65,270.65,270.65,214.65],atol=1e-10)
    np.testing.assert_allclose(P,[101325.,22632.06,5474.889,868.0187,110.9063,66.93887,3.956420],rtol=3e-6)
    np.testing.assert_allclose(jax.jit(jax.vmap(atm.pressure_to_standard_height))(P),z,atol=2e-10)
    # Continuity either side of interfaces and stable derivative in each layer.
    for h in H[1:]:
        a = atm.standard_temperature_pressure(geo.geopotential_to_geometric(h-.001))
        b = atm.standard_temperature_pressure(geo.geopotential_to_geometric(h+.001))
        np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-5)
    assert np.all(np.isnan(atm.standard_temperature_pressure(81000.)))
    assert np.isnan(atm.pressure_to_standard_height(0.))
    grad = jax.grad(lambda z:atm.standard_temperature_pressure(z)[1])(5000.)
    dz=.1
    finite = (atm.standard_temperature_pressure(5000+dz)[1]-atm.standard_temperature_pressure(5000-dz)[1])/(2*dz)
    assert float(grad) == pytest.approx(float(finite),rel=1e-8)


def test_dry_and_moist_air():
    air = atm.air_properties(288.15,101325.)
    assert bool(air.valid)
    assert float(air.density_kg_m3) == pytest.approx(1.225,rel=2e-6)
    assert float(air.sound_speed_m_s) == pytest.approx(340.294,rel=2e-6)
    assert float(air.dynamic_viscosity_Pa_s) == pytest.approx(1.78938e-5,rel=1e-5)
    moist = atm.air_properties(293.15,101325.,.5)
    pv = .5*611*10**(7.5*20/(237.3+20))
    expected = (101325-pv)/(atm.R_DRY*293.15)+pv/(atm.R_VAPOR*293.15)
    assert float(moist.density_kg_m3) == pytest.approx(expected)
    assert moist.density_kg_m3 < atm.air_properties(293.15,101325.).density_kg_m3
    assert np.isfinite(jax.grad(lambda T:atm.air_properties(T,20000.).density_kg_m3)(216.65))
    assert not bool(atm.air_properties(250.,90000.,.5).valid)
    assert not bool(atm.air_properties(293.,1000.,1.).valid)


def test_geographic_weather_and_flow():
    lla = jnp.array([.4,.5,1030.])
    wind = atm.meteorological_wind(10.,0.,2.)
    np.testing.assert_allclose(wind,[-10,0,-2])
    sample = jax.jit(lambda x:atm.atmosphere_at_lla(x,geoid_undulation_m=30.,wind_ned_m_s=wind))(lla)
    assert sample.air.temperature_K == atm.standard_temperature_pressure(1000.)[0]
    assert float(atm.approximate_gravity(0.,0.)) == pytest.approx(9.7803253359)
    assert float(atm.approximate_gravity(jnp.pi/2,0.)) == pytest.approx(9.8321849378,rel=1e-10)
    data,mach,reynolds,qbar = atm.flow_conditions(sample,jnp.array([100.,0,0]),jnp.eye(3),2.)
    assert float(data.speed_m_s) == pytest.approx(np.hypot(110.,2.))
    assert mach>0 and reynolds>0 and qbar>0


def test_weather_grid_time_interpolation_and_outside():
    axes = ([0.,1.],[0.,1.],[0.,2000.],[0.,10.])
    values = np.empty((2,2,2,2,6))
    for i,j,k,t in np.ndindex(2,2,2,2):
        values[i,j,k,t] = [290+2*i-4*k,100000-20000*k,.3,10*t,2*j,0]
    grid = regular_grid(axes,values)
    geoid = regular_grid(axes[:2],np.full((2,2),30.))
    fn = jax.jit(lambda lla,t:atm.atmosphere_from_grid(lla,grid,geoid,time_s=t))
    sample = fn(jnp.array([.5,.5,1030.]),5.)
    np.testing.assert_allclose([sample.air.temperature_K,sample.air.pressure_Pa],[289,90000])
    np.testing.assert_allclose(sample.wind_ned_m_s,[5,1,0])
    assert not bool(fn(jnp.array([.5,.5,1030.]),11.).air.valid)


def test_atmosphere_batched_dtype():
    query = jnp.array([0.,1000.,5000.],jnp.float32)
    samples = jax.jit(jax.vmap(lambda z:atm.air_properties(*atm.standard_temperature_pressure(z))))(query)
    assert samples.density_kg_m3.dtype == jnp.float32
    assert np.all(samples.valid)
