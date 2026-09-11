"""Explicit numerical grids, host validation and pure JAX multilinear lookup."""
from typing import NamedTuple, Any
import itertools
import numpy as np
import jax.numpy as jnp


class RegularGrid(NamedTuple):
    axes: tuple
    values: Any


def regular_grid(axes, values):
    """1..4 increasing axes; trailing value dimensions are independent channels.

    No implicit longitude wrapping, missing-data filling or extrapolation.
    Keep grid provenance, units and datum in the enclosing asset manifest.
    """
    axes = tuple(np.asarray(a, dtype=float) for a in axes)
    values = np.asarray(values, dtype=float)
    if not 1 <= len(axes) <= 4:
        raise ValueError("one to four axes required")
    if any(a.ndim != 1 or len(a)<2 or not np.all(np.isfinite(a)) or np.any(np.diff(a)<=0) for a in axes):
        raise ValueError("axes must be finite strictly increasing vectors of length >=2")
    if values.shape[:len(axes)] != tuple(len(a) for a in axes) or not np.all(np.isfinite(values)):
        raise ValueError("grid values must be finite and match axis dimensions")
    return RegularGrid(tuple(map(jnp.asarray,axes)),jnp.asarray(values))


def lookup(grid, point):
    """Single point, closed domain; out-of-grid -> NaN, never endpoint clamping."""
    indices, fractions, valid = [], [], jnp.asarray(True)
    for axis, x in zip(grid.axes,point,strict=True):
        i = jnp.clip(jnp.searchsorted(axis,x,side="right")-1,0,len(axis)-2)
        indices.append(i)
        fractions.append((x-axis[i])/(axis[i+1]-axis[i]))
        valid = valid & (x>=axis[0]) & (x<=axis[-1])
    result = jnp.zeros(grid.values.shape[len(grid.axes):],grid.values.dtype)
    for corner in itertools.product((0,1),repeat=len(grid.axes)):
        weight = jnp.asarray(1.,grid.values.dtype)
        for bit, fraction in zip(corner,fractions,strict=True):
            weight = weight*(fraction if bit else 1-fraction)
        result = result + weight*grid.values[tuple(i+b for i,b in zip(indices,corner,strict=True))]
    return jnp.where(valid,result,jnp.nan)
