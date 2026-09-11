"""guidance.slew: rate-limited pitch reference generator."""
import jax.numpy as jnp
from aerodrome.core.signals import PitchReference


def update(previous, navigation, goal, max_rate_rad_s, dt_s, tick):
    """A minimal guidance block. Navigation is unused for scheduled references.

    dt_s is the guidance period. Reference is held between updates; the
    timestamp records its generation tick, not its hypothetical future time.
    """
    increment = jnp.clip(goal.pitch_rad - previous.pitch_rad,
                         -max_rate_rad_s * dt_s, max_rate_rad_s * dt_s)
    return PitchReference(previous.pitch_rad + increment, tick)
