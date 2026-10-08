"""Full F-16 airframe entity for World and batched, compiled rollouts."""
from dataclasses import dataclass

from aerodrome.models.f16 import step
from .contracts import SystemSpec


@dataclass(frozen=True)
class F16Assembly:
    backend: str = "jax"
    initial_signals: tuple = ()
    systems: tuple = (SystemSpec("f16", "continuous", state_slots=("body",)),)

    def initialize(self, initial, key):
        return initial

    def make_tick(self, schedule):
        def tick(state, inputs, parameters, context):
            return step(state, inputs, context.resources, parameters, context.dt_s), state
        return tick
