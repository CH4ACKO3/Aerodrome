"""Engineering lower-atmosphere calculator; SI, Kelvin, RH fraction [0,1].

Standard atmosphere is a reference profile, not a location-based forecast.
Moist air: ideal mixture, constant heat capacities; dry-air Sutherland viscosity.
"""
from typing import NamedTuple, Any
import math
import jax.numpy as jnp
from aerodrome.core.geodesy import geometric_to_geopotential, geopotential_to_geometric, height_from_geoid_grid
from aerodrome.core.lookup import lookup
from aerodrome.core.airdata import air_relative_velocity, airdata, dynamic_pressure

R_DRY = 8314.32/28.9644  # US1976 universal gas constant / sea-level molar mass
R_VAPOR = 461.5
G0 = 9.80665
_HEIGHTS = (0.,11000.,20000.,32000.,47000.,51000.,71000.,84852.)
_LAPSES = (-.0065,0.,.001,.0028,0.,-.0028,-.002)
_TEMPS, _PRESSURES = [288.15], [101325.]
for _i, _lapse in enumerate(_LAPSES):
    _delta = _HEIGHTS[_i+1]-_HEIGHTS[_i]
    _t = _TEMPS[-1]+_lapse*_delta
    _p = (_PRESSURES[-1]*math.exp(-G0*_delta/(R_DRY*_TEMPS[-1])) if _lapse==0
          else _PRESSURES[-1]*(_t/_TEMPS[-1])**(-G0/(R_DRY*_lapse)))
    _TEMPS.append(_t)
    _PRESSURES.append(_p)


class AirProperties(NamedTuple):
    temperature_K: Any
    pressure_Pa: Any
    density_kg_m3: Any
    sound_speed_m_s: Any
    dynamic_viscosity_Pa_s: Any
    kinematic_viscosity_m2_s: Any
    relative_humidity: Any
    valid: Any


class AtmosphereSample(NamedTuple):
    air: AirProperties
    wind_ned_m_s: Any
    gravity_m_s2: Any


def standard_temperature_pressure(geometric_height_m):
    """US1976 lower-atmosphere layers, geometric -1..80 km.

    Negative heights extend the first lapse layer; above 80 km rejected.
    Values are molecular-scale approximations before composition changes.
    """
    H = geometric_to_geopotential(jnp.asarray(geometric_height_m))
    array = lambda x:jnp.asarray(x,dtype=H.dtype)
    index = jnp.clip(jnp.searchsorted(array(_HEIGHTS),H,side="right")-1,0,6)
    lapse = array(_LAPSES)[index]
    T0,P0 = array(_TEMPS)[index],array(_PRESSURES)[index]
    delta = H-array(_HEIGHTS)[index]
    T = T0+lapse*delta
    safe_lapse = jnp.where(lapse!=0,lapse,1.)
    P = P0*jnp.where(lapse==0,jnp.exp(-G0*delta/(R_DRY*T0)),
                     (T/T0)**(-G0/(R_DRY*safe_lapse)))
    valid = (geometric_height_m>=-1000)&(geometric_height_m<=80000)
    return jnp.where(valid,T,jnp.nan),jnp.where(valid,P,jnp.nan)


def saturation_vapor_pressure(temperature_K):
    """Tetens liquid-water saturation fit (Pa), restricted to 0..50 Celsius."""
    C = temperature_K-273.15
    value = 611.*10**(7.5*C/(237.3+C))
    return jnp.where((C>=0)&(C<=50),value,jnp.nan)


def pressure_to_standard_height(pressure_Pa):
    """Inverse ISA pressure -> geometric height; not measured true altitude."""
    pressure_Pa = jnp.asarray(pressure_Pa,dtype=jnp.result_type(pressure_Pa,1.))
    array = lambda x:jnp.asarray(x,dtype=pressure_Pa.dtype)
    index = jnp.clip(jnp.searchsorted(-array(_PRESSURES),-pressure_Pa,side="right")-1,0,6)
    T0,P0 = array(_TEMPS)[index],array(_PRESSURES)[index]
    lapse = array(_LAPSES)[index]
    safe_lapse = jnp.where(lapse!=0,lapse,1.)
    delta = jnp.where(lapse==0,-R_DRY*T0/G0*jnp.log(pressure_Pa/P0),
                      T0/safe_lapse*((pressure_Pa/P0)**(-R_DRY*lapse/G0)-1))
    height = geopotential_to_geometric(array(_HEIGHTS)[index]+delta)
    pmin = standard_temperature_pressure(array(80000.))[1]
    pmax = standard_temperature_pressure(array(-1000.))[1]
    return jnp.where((pressure_Pa>=pmin)&(pressure_Pa<=pmax),height,jnp.nan)


def air_properties(temperature_K, pressure_Pa, relative_humidity=0.):
    """Measured/local T,p,RH -> properties. Dry range 150..350 K.

    Nonzero humidity only 273.15..323.15 K (liquid saturation approximation).
    No silent clamping of supersaturated or physically invalid inputs.
    """
    dtype = jnp.result_type(temperature_K,pressure_Pa,relative_humidity,1.)
    T,P,rh = (jnp.asarray(x,dtype=dtype) for x in (temperature_K,pressure_Pa,relative_humidity))
    # Finite inactive saturation branch permits differentiating dry cold air.
    saturation = saturation_vapor_pressure(jnp.clip(T,273.15,323.15))
    vapor = rh*saturation
    dry_density = (P-vapor)/(R_DRY*T)
    vapor_density = vapor/(R_VAPOR*T)
    rho = dry_density+vapor_density
    mixing = vapor_density/rho
    gas_constant = (1-mixing)*R_DRY+mixing*R_VAPOR
    cp = (1-mixing)*(3.5*R_DRY)+mixing*1859.
    gamma = cp/(cp-gas_constant)
    sound = jnp.sqrt(gamma*gas_constant*T)
    mu = 1.458e-6*T**1.5/(T+110.4)
    valid = ((T>=150)&(T<=350)&(P>0)&(rh>=0)&(rh<=1)&(vapor<P)
             &jnp.isfinite(P)&((rh==0)|((T>=273.15)&(T<=323.15))))
    values = [jnp.where(valid,x,jnp.nan) for x in (T,P,rho,sound,mu,mu/rho,rh)]
    return AirProperties(*values,valid)


def approximate_gravity(latitude_rad, geometric_height_m):
    """Somigliana surface gravity + spherical inverse-square height correction.

    Engineering approximation, not an EGM gravity field or ECEF gravity vector.
    """
    s2 = jnp.sin(latitude_rad)**2
    surface = 9.7803253359*(1+.00193185265241*s2)/jnp.sqrt(1-.00669437999013*s2)
    value = surface*(6371000./(6371000.+geometric_height_m))**2
    return jnp.where((jnp.abs(latitude_rad)<=jnp.pi/2)&(geometric_height_m>=-1000)&(geometric_height_m<=80000),value,jnp.nan)


def meteorological_wind(speed_m_s, from_bearing_rad, upward_m_s=0.):
    """Wind FROM clockwise-from-North bearing -> air-mass velocity in NED."""
    wind = jnp.stack((-speed_m_s*jnp.cos(from_bearing_rad),-speed_m_s*jnp.sin(from_bearing_rad),-upward_m_s))
    return jnp.where(speed_m_s>=0,wind,jnp.nan)


def atmosphere_at_lla(lla, *, geoid_undulation_m, temperature_offset_K=0.,
                      pressure_Pa=None, relative_humidity=0., wind_ned_m_s=None):
    """LLA ellipsoid height -> h-N treated as geometric altitude for ISA.

    Explicit geoid required (zero is an explicit approximation). Temperature
    offsets retain standard pressure unless supplied; not a new hydrostatic
    column. Latitude affects approximate gravity, not invented weather.
    """
    height = lla[2]-geoid_undulation_m
    T,P = standard_temperature_pressure(height)
    air = air_properties(T+temperature_offset_K,P if pressure_Pa is None else pressure_Pa,relative_humidity)
    wind = jnp.zeros(3,dtype=lla.dtype) if wind_ned_m_s is None else wind_ned_m_s
    return AtmosphereSample(air,wind,approximate_gravity(lla[0],height))


def atmosphere_from_grid(lla, weather_grid, geoid_grid, *, time_s=None):
    """Grid axes [lat,lon,orthometric height,(time)]; channels [T,P,RH,wN,wE,wD].

    Out-of-domain values propagate NaN. Caller owns datum, epoch and seam
    preparation. P is interpolated linearly, not logarithmically.
    """
    height = height_from_geoid_grid(lla,geoid_grid)
    point = (lla[0],lla[1],height) if time_s is None else (lla[0],lla[1],height,time_s)
    values = lookup(weather_grid,point)
    if values.shape != (6,):
        raise ValueError("weather grid must have six channels: T,P,RH,wN,wE,wD")
    return AtmosphereSample(air_properties(*values[:3]),values[3:],approximate_gravity(lla[0],height))


def flow_conditions(sample, velocity_body_m_s, rotation_nb, reference_length_m):
    """Return (AirData, Mach, Reynolds, dynamic pressure); includes wind."""
    data = airdata(air_relative_velocity(velocity_body_m_s,sample.wind_ned_m_s,rotation_nb))
    a = sample.air
    reynolds = a.density_kg_m3*data.speed_m_s*reference_length_m/a.dynamic_viscosity_Pa_s
    return (data,data.speed_m_s/a.sound_speed_m_s,
            jnp.where(reference_length_m>0,reynolds,jnp.nan),dynamic_pressure(a.density_kg_m3,data.speed_m_s))
