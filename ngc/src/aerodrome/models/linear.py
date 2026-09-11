"""Discrete LTI numerical kernels. No control/SciPy imports or host conversions."""
from typing import NamedTuple, Any
import jax


class LinearParameters(NamedTuple):
    A: Any
    B: Any
    C: Any
    D: Any


def output(state, inputs, parameters):
    """y[k] = C x[k] + D u[k]; output precedes the state update."""
    return parameters.C @ state + parameters.D @ inputs


def step(state, inputs, parameters):
    """Zero-order-held input. Caller owns sampling period and parameter axes."""
    return parameters.A @ state + parameters.B @ inputs, output(state, inputs, parameters)


def rollout(state, inputs, parameters):
    """Single-system time-major inputs [T,nu]; compose with jit/vmap as needed."""
    return jax.lax.scan(lambda x,u: step(x,u,parameters), state, inputs)
