"""Integrators own continuous state; evaluate coupled derivatives at every stage."""
import jax


def rk4(rhs, time_s, state, inputs, parameters, dt_s):
    """rhs(t, PyTree state, held inputs, parameters) -> derivative PyTree."""
    add = lambda x, k, scale: jax.tree.map(lambda a, b: a + scale * b, x, k)
    k1 = rhs(time_s, state, inputs, parameters)
    k2 = rhs(time_s + dt_s / 2, add(state, k1, dt_s / 2), inputs, parameters)
    k3 = rhs(time_s + dt_s / 2, add(state, k2, dt_s / 2), inputs, parameters)
    k4 = rhs(time_s + dt_s, add(state, k3, dt_s), inputs, parameters)
    return jax.tree.map(lambda x, a, b, c, d: x + dt_s * (a + 2*b + 2*c + d) / 6,
                        state, k1, k2, k3, k4)
