"""Generic rigid-body entity for World, BatchedWorld and compiled rollouts."""
from dataclasses import dataclass
from typing import Callable
from aerodrome.models.rigid_body import RigidBody6DoF, held_loads
from .contracts import SystemSpec


@dataclass(frozen=True)
class RigidBodyAssembly:
    model: RigidBody6DoF = RigidBody6DoF()
    load_fn: Callable = held_loads
    backend: str = "jax"
    initial_signals: tuple = ()
    systems: tuple = (SystemSpec("rigid_body", "continuous", state_slots=("body",)),)

    def initialize(self, initial, key):
        # Initial conditions come from model.initialize; no stochastic state here.
        return initial

    def make_tick(self, schedule):
        def tick(state, inputs, parameters, context):
            following = self.model.step(state, inputs, parameters, context.dt_s,
                                        time_s=context.tick*context.dt_s,
                                        load_fn=self.load_fn, resources=context.resources)
            return following, state  # Beginning-of-interval convention used by World.
        return tick
