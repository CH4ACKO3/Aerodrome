"""Lower independent native entities into separate temporal task chains."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from aerodrome.composition.world import TickContext, WorldState, WorldRecord
from .dataflow import Task, TaskGraph, DependencyExecutor


@dataclass(frozen=True)
class WorldGraphPlan:
    graph: TaskGraph
    chains: tuple[tuple, ...]
    start_tick: object
    steps: int
    total_ticks: int


class GraphWorldRunner:
    """A finite fixed-input rollout; no global barrier between entity chunks.

    Current World entities are independent. This adapter does not infer dependencies
    from Python functions or change the internal equations of an Assembly.
    """

    def __init__(self, world, *, max_workers=2, chunk_ticks=32):
        if type(chunk_ticks) is not int or chunk_ticks < 1:
            raise ValueError("chunk_ticks must be a positive integer")
        self.world, self.chunk_ticks = world, chunk_ticks
        self.executor = DependencyExecutor(max_workers=max_workers)
        self._chunks = {}

    def _chunk(self, entity_index, length):
        key = (entity_index, length)
        if key not in self._chunks:
            tick = self.world._ticks[entity_index]
            dt = self.world.spec.schedule.physics_dt_s

            def advance(state, inputs, parameters, resources, start_tick):
                def body(carry, _):
                    state, k = carry
                    following, record = tick(state, inputs, parameters, TickContext(k, dt, resources))
                    return (following, k + 1), record
                (following, _), records = jax.lax.scan(body, (state, start_tick), None, length)
                return following, records

            self._chunks[key] = jax.jit(advance)
        return self._chunks[key]

    def plan(self, state, inputs, parameters, *, steps):
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        self.world.validate(state, inputs, parameters)
        total = steps * self.world.spec.ticks_per_step
        tasks, chains = [], []
        for index, entity_id in enumerate(self.world.entity_ids):
            chain = []
            for start in range(0, total, self.chunk_ticks):
                key = (entity_id, start)  # tick offset from the supplied state
                parent = chain[-1] if chain else None
                length = min(self.chunk_ticks, total - start)
                advance = self._chunk(index, length)

                def run(dependencies, *, index=index, parent=parent, start=start, advance=advance):
                    previous = state.entities[index] if parent is None else dependencies[parent][0]
                    return advance(previous, inputs[index], parameters.entities[index],
                                   parameters.resources, state.tick + start)

                tasks.append(Task(key, run, () if parent is None else (parent,)))
                chain.append(key)
            chains.append(tuple(chain))
        return WorldGraphPlan(TaskGraph(tasks), tuple(chains), state.tick, steps, total)

    def run(self, state, inputs, parameters, *, steps, on_complete=None):
        plan = self.plan(state, inputs, parameters, steps=steps)
        results = self.executor.run(plan.graph, on_complete=on_complete)
        final = WorldState(plan.start_tick + plan.total_ticks,
                           tuple(results[chain[-1]][0] for chain in plan.chains))
        traces = []
        for chain in plan.chains:
            joined = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0),
                                  *(results[key][1] for key in chain))
            traces.append(jax.tree.map(
                lambda x: x.reshape((steps, self.world.spec.ticks_per_step) + x.shape[1:]), joined))
        ticks = (plan.start_tick + jnp.arange(plan.total_ticks, dtype=state.tick.dtype)).reshape(
            steps, self.world.spec.ticks_per_step)
        record = WorldRecord(ticks, ticks * self.world.spec.schedule.physics_dt_s, tuple(traces))
        return jax.block_until_ready((final, record))
