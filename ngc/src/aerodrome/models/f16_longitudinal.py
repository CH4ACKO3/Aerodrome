"""Symmetric longitudinal reduction of ISRL/NASA1538 F16 aerodynamics.

Ported coefficient subset from F16AeroFM.m, copyright 2023 Raktim Bhattacharya,
MIT; see data/f16/LICENSE. Planar equations/atmosphere/ideal actuators are explicit
Aerodrome assumptions. No claim of a certified or full-envelope F16 model.
"""
import hashlib
import json
from pathlib import Path
from typing import NamedTuple,Any
import itertools
import jax
import jax.numpy as jnp
import numpy as np


class Airframe(NamedTuple):
    mass: Any = 636.94*14.5939
    iyy: Any = 55814.*14.5939*.3048**2
    area: Any = 300*.3048**2
    chord: Any = 11.32*.3048
    xcg: Any = .30
    xcgr: Any = .35
    gravity: Any = 9.806
    lef_rad: Any = 0.


def load_tables(directory=None):
    folder = Path(directory) if directory else Path(__file__).parent/"data"/"f16"
    manifest = json.loads((folder/"manifest.json").read_text())
    path = folder/"longitudinal.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["npz_sha256"]:
        raise ValueError("F16 data integrity check failed")
    with np.load(path,allow_pickle=False) as data:
        return {k:jnp.asarray(data[k]) for k in data.files}


def interpolate(axes,values,point):
    """Multilinear interpolation with NaN outside grid, including endpoint support."""
    indices,weights,valid = [],[],jnp.asarray(True)
    for axis,x in zip(axes,point,strict=True):
        index = jnp.clip(jnp.searchsorted(axis,x,side="right")-1,0,len(axis)-2)
        indices.append(index)
        weights.append((x-axis[index])/(axis[index+1]-axis[index]))
        valid = valid & (x >= axis[0]) & (x <= axis[-1])
    result = jnp.zeros((),values.dtype)
    for corner in itertools.product((0,1),repeat=len(axes)):
        weight = jnp.asarray(1.,values.dtype)
        for bit,fraction in zip(corner,weights,strict=True):
            weight = weight*(fraction if bit else 1-fraction)
        result = result+weight*values[tuple(i+b for i,b in zip(indices,corner,strict=True))]
    return jnp.where(valid,result,jnp.nan)


def density(height):
    """NASA Glenn metric troposphere fit, only 0..11000 m here."""
    temperature = 15.04-.00649*height
    pressure_kpa = 101.29*((temperature+273.1)/288.08)**5.256
    rho = pressure_kpa/(.2869*(temperature+273.1))
    return jnp.where((height >= 0)&(height <= 11000),rho,jnp.nan)


def coefficients(speed,alpha,q,elevator,tables,p=Airframe()):
    a,e = jnp.rad2deg(alpha),jnp.rad2deg(elevator)
    a1,a2,dh = tables["alpha1"],tables["alpha2"],tables["dh1"]
    def base(name,angle=e):
        return interpolate((a1,dh),tables[name],(a,angle))
    def one(name,axis=a1,x=a):
        return interpolate((axis,),tables[name],(x,))
    lef = 1-jnp.rad2deg(p.lef_rad)/25
    Cx = base("Cx")+(one("Cx_lef",a2)-base("Cx",0.))*lef+p.chord/(2*speed)*(one("Cxq")+one("deltaCxq_lef",a2)*lef)*q
    Cz = base("Cz")+(one("Cz_lef",a2)-base("Cz",0.))*lef+p.chord/(2*speed)*(one("Czq")+one("deltaCzq_lef",a2)*lef)*q
    Cm = base("Cm")*one("eta_el",dh,e)+Cz*(p.xcgr-p.xcg)+(one("Cm_lef",a2)-base("Cm",0.))*lef+p.chord/(2*speed)*(one("Cmq")+one("deltaCmq_lef",a2)*lef)*q+one("deltaCm")
    return jnp.stack((Cx,Cz,Cm))


def rhs(state,inputs,tables,p=Airframe()):
    """x=[V,alpha,q,theta,h] SI; u=[elevator_rad,thrust_N]. z-body positive down."""
    speed,alpha,q,theta,height = state
    elevator,thrust = inputs
    cx,cz,cm = coefficients(speed,alpha,q,elevator,tables,p)
    scale = .5*density(height)*speed**2*p.area
    X,Z,M = scale*cx+thrust,scale*cz,scale*p.chord*cm
    gamma = theta-alpha
    return jnp.stack(((X*jnp.cos(alpha)+Z*jnp.sin(alpha))/p.mass-p.gravity*jnp.sin(gamma),
                      q+(Z*jnp.cos(alpha)-X*jnp.sin(alpha))/(p.mass*speed)+p.gravity*jnp.cos(gamma)/speed,
                      M/p.iyy,q,speed*jnp.sin(gamma)))
