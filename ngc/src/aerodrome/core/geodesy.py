"""WGS84 geodesy, radians/metres. LLA = [geodetic latitude, longitude, h].

h is ellipsoidal height, not height above sea level. Use float64 for ECEF.
Single-point kernels support jit/vmap. Earth rotation is not modelled here.
"""
import jax
import jax.numpy as jnp
from .frames import ecef_to_ned_matrix
from .units import wrap_pi
from .lookup import lookup

WGS84_A_M = 6378137.
WGS84_F = 1/298.257223563
WGS84_E2 = WGS84_F*(2-WGS84_F)
WGS84_B_M = WGS84_A_M*(1-WGS84_F)
STANDARD_EARTH_RADIUS_M = 6356766.


def geodetic_to_ecef(lla):
    lat, lon, h = lla
    s,c = jnp.sin(lat),jnp.cos(lat)
    N = WGS84_A_M/jnp.sqrt(1-WGS84_E2*s*s)
    xyz = jnp.stack(((N+h)*c*jnp.cos(lon),(N+h)*c*jnp.sin(lon),(N*(1-WGS84_E2)+h)*s))
    return jnp.where((jnp.abs(lat)<=jnp.pi/2)&jnp.all(jnp.isfinite(lla))&(h>=-1000.),xyz,jnp.nan)


def ecef_to_geodetic(xyz):
    """Fixed 10-iteration inverse, surface/near-Earth use (h >= -1 km).

    Pole longitude is conventionally zero. Interior/centre inputs -> NaN.
    The polar coordinate singularity is not a differentiable chart.
    """
    x,y,z = xyz
    p = jnp.hypot(x,y)
    polar = p < 1e-8
    safe_p = jnp.where(polar,1.,p)
    lat = jnp.arctan2(z,safe_p*(1-WGS84_E2))
    def update(_, lat):
        N = WGS84_A_M/jnp.sqrt(1-WGS84_E2*jnp.sin(lat)**2)
        return jnp.arctan2(z+WGS84_E2*N*jnp.sin(lat),safe_p)
    lat = jax.lax.fori_loop(0,10,update,lat)
    lat = jnp.where(polar,jnp.sign(z)*jnp.pi/2,lat)
    lon = jnp.arctan2(jnp.where(polar,0.,y),jnp.where(polar,1.,x))
    # Projection onto the ellipsoid normal avoids division by cos(lat).
    h = p*jnp.cos(lat)+z*jnp.sin(lat)-WGS84_A_M*jnp.sqrt(1-WGS84_E2*jnp.sin(lat)**2)
    result = jnp.stack((lat,lon,h))
    return jnp.where(jnp.all(jnp.isfinite(xyz))&(h>=-1000.00001),result,jnp.nan)


def geodetic_to_ned(lla, origin_lla):
    return ecef_to_ned_matrix(*origin_lla[:2])@(geodetic_to_ecef(lla)-geodetic_to_ecef(origin_lla))


def ned_to_geodetic(offset_ned_m, origin_lla):
    xyz = geodetic_to_ecef(origin_lla)+ecef_to_ned_matrix(*origin_lla[:2]).T@offset_ned_m
    return ecef_to_geodetic(xyz)


def geocentric_radius(latitude_rad, ellipsoid_height_m=0.):
    """Distance to Earth's centre at a given GEODETIC latitude and height."""
    return jnp.linalg.norm(geodetic_to_ecef(jnp.stack((latitude_rad,jnp.zeros_like(latitude_rad),ellipsoid_height_m))))


def radius_to_ellipsoid_height(radius_m, latitude_rad):
    """Inverse needs geodetic latitude: no universal radius minus Earth radius."""
    s,c = jnp.sin(latitude_rad),jnp.cos(latitude_rad)
    N = WGS84_A_M/jnp.sqrt(1-WGS84_E2*s*s)
    projection = N*(1-WGS84_E2*s*s)
    radius0_sq = (N*c)**2+(N*(1-WGS84_E2)*s)**2
    discriminant = projection**2+radius_m**2-radius0_sq
    h = (radius_m**2-radius0_sq)/(jnp.sqrt(jnp.maximum(discriminant,0.))+projection)
    return jnp.where((radius_m>0)&(discriminant>=0)&(jnp.abs(latitude_rad)<=jnp.pi/2)&(h>=-1000.00001),h,jnp.nan)


def ellipsoid_to_orthometric(height_m, geoid_undulation_m):
    return height_m-geoid_undulation_m


def orthometric_to_ellipsoid(height_m, geoid_undulation_m):
    return height_m+geoid_undulation_m


def height_above_ground(orthometric_height_m, terrain_orthometric_m):
    return orthometric_height_m-terrain_orthometric_m


def height_from_geoid_grid(lla, geoid_grid):
    """h-N(lat,lon); grid must use the same horizontal/vertical datum."""
    return ellipsoid_to_orthometric(lla[2],lookup(geoid_grid,lla[:2]))


def geometric_to_geopotential(height_m):
    r = STANDARD_EARTH_RADIUS_M
    return jnp.where(height_m>-r,r*height_m/(r+height_m),jnp.nan)


def geopotential_to_geometric(height_m):
    r = STANDARD_EARTH_RADIUS_M
    return jnp.where(height_m<r,r*height_m/(r-height_m),jnp.nan)


def direction_offset(offset_ned, heading_rad):
    """[forward,right,down] relative to a horizontal reference heading."""
    n,e,d = offset_ned
    c,s = jnp.cos(heading_rad),jnp.sin(heading_rad)
    return jnp.stack((c*n+s*e,-s*n+c*e,d))


def relative_position(target_lla, origin_lla, heading_rad=0.):
    """Return (forward/right/down, azimuth error, elevation, slant range).

    Local tangent line-of-sight geometry, NOT a surface geodesic distance.
    Azimuth right-positive, elevation up-positive; undefined angles -> NaN.
    """
    v = direction_offset(geodetic_to_ned(target_lla,origin_lla),heading_rad)
    horizontal = jnp.hypot(v[0],v[1])
    distance = jnp.linalg.norm(v)
    az = jnp.where(horizontal>1e-8,wrap_pi(jnp.arctan2(v[1],v[0])),jnp.nan)
    el = jnp.where(distance>1e-8,jnp.arctan2(-v[2],horizontal),jnp.nan)
    return v,az,el,distance
