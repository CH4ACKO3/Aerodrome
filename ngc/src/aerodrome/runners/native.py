"""The same step function in inspectable Python and compiled JAX execution."""
import jax
import jax.numpy as jnp


def run_loop(step, initial, goal, params, steps):
    if steps < 1:
        raise ValueError("steps must be positive")
    state, records = initial, []
    for _ in range(steps):
        state, record = step(state, goal, params)
        records.append(record)
    return state, jax.tree.map(lambda *xs: jnp.stack(xs), *records)


def run_scan(step, initial, goal, params, steps):
    """steps is static at compilation; logs have fixed leading time dimension."""
    if steps < 1:
        raise ValueError("steps must be positive")
    return jax.lax.scan(lambda state, _: step(state, goal, params), initial, None, length=steps)
