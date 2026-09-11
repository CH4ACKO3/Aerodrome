"""Use an executable all-JAX module graph as an entity in the existing World."""
from dataclasses import dataclass
import jax.numpy as jnp
from .contracts import SystemSpec
from .module_graph import ModuleGraphState


@dataclass(frozen=True)
class GraphAssembly:
    graph: object
    initial_signals: tuple = ()

    @property
    def backend(self):
        return "jax" if all(m.backend == "jax" for m in self.graph.modules.values()) else "host"

    @property
    def systems(self):
        # Port-level wiring and delays were validated by ModuleGraph. These
        # entries expose per-module state ownership to the World inspection API.
        return tuple(SystemSpec(name, "discrete", state_slots=(name,),
                                equation=self.graph.modules[name].equation) for name in self.graph.order)

    def initialize(self, initial_conditions, key):
        # A factory can derive explicit module PRNG states from the entity key.
        values = initial_conditions(key) if callable(initial_conditions) else initial_conditions
        if set(values) != set(self.graph.modules):
            raise ValueError("initial values must cover all graph modules")
        return ModuleGraphState(jnp.asarray(0, jnp.int32), dict(values))

    def make_tick(self, schedule):
        tick = self.graph.make_tick(schedule.physics_dt_s)
        def advance(state, inputs, parameters, context):
            return tick(state._replace(tick=context.tick), inputs, parameters)
        return advance
