"""Static ECS-inspired world. State/parameters are PyTrees, topology is Python."""
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
from aerodrome.core.clock import Schedule
from .contracts import Assembly, validate_systems


@dataclass(frozen=True)
class EntitySpec:
    id: str
    assembly: Assembly


@dataclass(frozen=True)
class WorldSpec:
    entities: tuple[EntitySpec, ...]
    schedule: Schedule = Schedule()
    ticks_per_step: int = 2

    def __post_init__(self):
        object.__setattr__(self, "entities", tuple(self.entities))
        ids = [e.id for e in self.entities]
        if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("world requires unique, nonempty entity IDs")
        if type(self.ticks_per_step) is not int or self.ticks_per_step < 1:
            raise ValueError("ticks_per_step must be a positive integer")


class WorldState(NamedTuple):
    tick: Any
    entities: tuple


class WorldParameters(NamedTuple):
    entities: tuple
    resources: Any = ()  # shared numeric atmosphere/tables/etc.; no file handles


class TickContext(NamedTuple):
    tick: Any
    dt_s: Any
    resources: Any


class WorldRecord(NamedTuple):
    tick: Any
    time_s: Any
    entities: tuple


@dataclass(frozen=True)
class World:
    spec: WorldSpec
    _ticks: tuple
    _streams: tuple[int, ...]

    @property
    def entity_ids(self):
        return tuple(e.id for e in self.spec.entities)

    @property
    def step_dt_s(self):
        return self.spec.schedule.physics_dt_s * self.spec.ticks_per_step

    def pack(self, values):
        """Resolve names once at the host boundary, never rely on dict ordering."""
        if set(values) != set(self.entity_ids):
            raise ValueError(f"expected exactly these entity IDs: {self.entity_ids}")
        return tuple(values[name] for name in self.entity_ids)

    def parameters(self, entities, *, resources=()):
        return WorldParameters(self.pack(entities), resources)

    def reset(self, seed, initial_conditions):
        return self.reset_from_key(jax.random.key(seed), initial_conditions)

    def reset_from_key(self, root, initial_conditions):
        """Explicit key entry point for stable world/episode random namespaces."""
        conditions = self.pack(initial_conditions)
        states = tuple(e.assembly.initialize(c, jax.random.fold_in(root, stream))
                       for e, c, stream in zip(self.spec.entities, conditions, self._streams, strict=True))
        return WorldState(jnp.asarray(0, jnp.int32), states)

    def entity_state(self, state, entity_id):
        return state.entities[self.entity_ids.index(entity_id)]

    def tick(self, state, inputs, parameters):
        """One physics interval. Records describe its beginning, not its end."""
        count = len(self._ticks)
        if len(state.entities) != count or len(inputs) != count or len(parameters.entities) != count:
            raise ValueError("entity state/input/parameter count does not match WorldSpec")
        context = TickContext(state.tick, self.spec.schedule.physics_dt_s, parameters.resources)
        pairs = tuple(fn(s, u, p, context) for fn, s, u, p in
                      zip(self._ticks, state.entities, inputs, parameters.entities, strict=True))
        return (WorldState(state.tick + 1, tuple(pair[0] for pair in pairs)),
                WorldRecord(state.tick, state.tick * context.dt_s, tuple(pair[1] for pair in pairs)))

    def step(self, state, inputs, parameters):
        """One decision interval; hold inputs, return ALL constituent tick records."""
        return jax.lax.scan(lambda s, _: self.tick(s, inputs, parameters), state,
                            xs=None, length=self.spec.ticks_per_step)

    def rollout(self, state, inputs, parameters, *, steps):
        """Fixed input rollout. Trace axes are [decision_step, physics_tick, ...]."""
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive static integer")
        return jax.lax.scan(lambda s, _: self.step(s, inputs, parameters), state,
                            xs=None, length=steps)

    def validate(self, state, inputs, parameters):
        """Trace before a long run; check that tick preserves state shape/dtype."""
        next_state, _ = jax.eval_shape(self.tick, state, inputs, parameters)
        signature = lambda tree: (jax.tree.structure(tree),
                                  [(x.shape, x.dtype) for x in jax.tree.leaves(tree)])
        if signature(state) != signature(next_state):
            raise ValueError("assembly tick must preserve state PyTree, shapes and dtypes")


def build_world(spec, *, backend="jax"):
    if backend != "jax":
        raise ValueError("build_world supports jax; use CoSimulationRunner for host-owned sessions")
    streams = tuple(int.from_bytes(sha256(e.id.encode()).digest()[:4], "little") for e in spec.entities)
    if len(set(streams)) != len(streams):
        raise ValueError("entity random stream collision; choose different IDs")
    ticks = []
    for entity in spec.entities:
        if entity.assembly.backend != "jax":
            raise ValueError(f"{entity.id}: external components cannot run inside a native JAX world")
        validate_systems(entity.assembly.systems, entity.assembly.initial_signals)
        ticks.append(entity.assembly.make_tick(spec.schedule))
    return World(spec, tuple(ticks), streams)
