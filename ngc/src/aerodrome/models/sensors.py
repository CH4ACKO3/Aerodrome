"""sensor.pitch: y = theta + sigma*epsilon, epsilon ~ N(0,1)."""
import jax
import jax.numpy as jnp
from aerodrome.core.signals import PitchMeasurement


def measure_pitch(truth, key, tick, standard_deviation_rad, available):
    """One fresh sample only on sensor ticks; zero is an invalid placeholder.

    fold_in with source ID and tick prevents control scheduling from changing
    this sensor's noise sequence. Caller supplies a per-experiment root key.
    """
    source_key = jax.random.fold_in(key, 0)
    sample_key = jax.random.fold_in(source_key, tick)
    value = jax.lax.cond(
        available,
        lambda: truth.pitch_rad + standard_deviation_rad * jax.random.normal(
            sample_key, (), dtype=truth.pitch_rad.dtype),
        lambda: jnp.zeros_like(truth.pitch_rad),
    )
    return PitchMeasurement(value, tick, available)
