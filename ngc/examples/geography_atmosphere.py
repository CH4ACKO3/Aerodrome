"""Reproducible synthetic geographic scenario and standard-atmosphere tables."""
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
from aerodrome.core import geodesy as geo
from aerodrome.core.lookup import regular_grid,lookup
from aerodrome.models import atmosphere as atm


def main():
    jax.config.update("jax_enable_x64",True)
    folder = Path("artifacts/geography_atmosphere")
    folder.mkdir(parents=True,exist_ok=True)
    lat,lon = jnp.deg2rad(jnp.array([22.5,114.]))
    lla = jnp.array([lat,lon,3030.])
    # Synthetic N=30 m is an explicit example, not an EGM/DEM measurement.
    geoid = regular_grid(([lat-.1,lat+.1],[lon-.1,lon+.1]),np.full((2,2),30.))
    orthometric = geo.height_from_geoid_grid(lla,geoid)
    target = geo.ned_to_geodetic(jnp.array([1000.,200.,-100.]),lla)
    _,az,el,distance = geo.relative_position(target,lla,jnp.deg2rad(15.))
    wind = atm.meteorological_wind(12.,jnp.deg2rad(270.))
    sample = atm.atmosphere_at_lla(lla,geoid_undulation_m=lookup(geoid,lla[:2]),
                                  temperature_offset_K=10.,relative_humidity=.4,wind_ned_m_s=wind)
    data,mach,reynolds,qbar = atm.flow_conditions(sample,jnp.array([150.,0.,0.]),jnp.eye(3),3.)
    assert bool(sample.air.valid) and bool(data.angles_valid)
    heights = jnp.linspace(0.,80000.,161)
    air = jax.jit(jax.vmap(lambda h:atm.air_properties(*atm.standard_temperature_pressure(h))))(heights)
    radii = jax.vmap(lambda h:geo.geocentric_radius(lat,h))(heights)
    rows = np.column_stack((heights,geo.geometric_to_geopotential(heights),
                            air.temperature_K,air.pressure_Pa,air.density_kg_m3,air.sound_speed_m_s))
    np.savetxt(folder/"standard_atmosphere.csv",rows,delimiter=",",comments="",
               header="geometric_height_m,geopotential_height_m,temperature_K,pressure_Pa,density_kg_m3,sound_speed_m_s")
    np.savetxt(folder/"radius_at_22p5deg.csv",np.column_stack((heights,radii)),delimiter=",",comments="",
               header="ellipsoid_height_m,geocentric_radius_m")
    np.savez_compressed(folder/"reference_table.npz",geometric_height_m=heights,
                        ellipsoid_height_m=heights,latitude_rad=lat,radius_m=radii,
                        temperature_K=air.temperature_K,pressure_Pa=air.pressure_Pa)
    summary = dict(synthetic_geoid_undulation_m=30.,orthometric_height_m=float(orthometric),
                   ecef_m=np.asarray(geo.geodetic_to_ecef(lla)).tolist(),
                   target_azimuth_error_deg=float(jnp.rad2deg(az)),target_elevation_deg=float(jnp.rad2deg(el)),
                   target_slant_range_m=float(distance),temperature_K=float(sample.air.temperature_K),
                   pressure_Pa=float(sample.air.pressure_Pa),density_kg_m3=float(sample.air.density_kg_m3),
                   wind_ned_m_s=np.asarray(wind).tolist(),mach=float(mach),reynolds=float(reynolds),
                   dynamic_pressure_Pa=float(qbar),backend=jax.default_backend())
    (folder/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
