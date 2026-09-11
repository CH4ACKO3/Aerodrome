"""Lower an executable module graph into dependency-safe compiled temporal regions.

Finite event DAG. Fuse connected JAX events only if they share a chunk window and
the EXACT set of host ancestors. Thus fusion cannot create a host exit/reentry cycle.
"""
from dataclasses import dataclass
import math
import numpy as np
import jax
import jax.numpy as jnp
from aerodrome.composition.module_graph import (
    InputBinding, ModuleContext, ModuleGraphState, ModuleGraphTrace, check_outputs,
)
from .dataflow import Task, TaskGraph, DependencyExecutor


@dataclass(frozen=True)
class Event:
    key: tuple[str, int]
    previous: object
    inputs: tuple  # (port_name, event_key or None, source_module, source_port, external_name)
    dependencies: tuple


@dataclass(frozen=True)
class Region:
    id: int
    backend: str
    events: tuple
    dependencies: tuple[int, ...]
    host_ancestors: frozenset


class CompiledModuleGraph:
    def __init__(self, graph, *, ticks, physics_dt_s, chunk_ticks=32, start_tick=0, fuse=True):
        for name, number, minimum in (("ticks", ticks, 1), ("chunk_ticks", chunk_ticks, 1),
                                       ("start_tick", start_tick, 0)):
            if type(number) is not int or number < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(physics_dt_s) or physics_dt_s <= 0:
            raise ValueError("physics_dt_s must be finite and positive")
        self.graph, self.ticks, self.dt = graph, ticks, physics_dt_s
        self.start_tick, self.chunk_ticks = start_tick, chunk_ticks
        events = []

        def source_event(module_id, k):
            period = graph.modules[module_id].every
            sampled = k - k % period
            return (module_id, sampled) if sampled >= start_tick else None

        self._source_event = source_event
        for k in range(start_tick, start_tick + ticks):
            for name in graph.order:
                module = graph.modules[name]
                if k % module.every:
                    continue
                previous = source_event(name, k - 1)
                dependencies = set(() if previous is None else (previous,))
                ports = []
                for port in module.inputs:
                    source = graph.sources[(name, port.name)]
                    if isinstance(source, InputBinding):
                        ports.append((port.name, None, None, None, source.source))
                    else:
                        producer = source_event(source.source[0], k - source.delay)
                        if producer is not None:
                            dependencies.add(producer)
                        ports.append((port.name, producer, *source.source, None))
                events.append(Event((name, k), previous, tuple(ports), tuple(sorted(dependencies))))
        self.events = tuple(events)
        self._events = {event.key: event for event in events}
        ancestors, parent = {}, {event.key: event.key for event in events}
        for event in events:
            hosts = set()
            for dependency in event.dependencies:
                if dependency not in ancestors:
                    raise ValueError("event graph contains a future or unscheduled dependency")
                hosts.update(ancestors[dependency])
                if graph.modules[dependency[0]].backend == "host":
                    hosts.add(dependency)
            ancestors[event.key] = frozenset(hosts)

        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        if fuse:
            for event in events:
                if graph.modules[event.key[0]].backend != "jax":
                    continue
                for dependency in event.dependencies:
                    if (graph.modules[dependency[0]].backend == "jax" and
                            ancestors[event.key] == ancestors[dependency] and
                            (event.key[1]-start_tick)//chunk_ticks == (dependency[1]-start_tick)//chunk_ticks):
                        parent[find(event.key)] = find(dependency)
        groups = {}
        for event in events:
            groups.setdefault(find(event.key), []).append(event.key)
        locations = {key: (i, j) for i, group in enumerate(groups.values()) for j, key in enumerate(group)}
        regions = []
        for i, keys in enumerate(groups.values()):
            dependencies = {locations[d][0] for key in keys for d in self._events[key].dependencies
                            if locations[d][0] != i}
            regions.append(Region(i, graph.modules[keys[0][0]].backend, tuple(keys),
                                  tuple(sorted(dependencies)), ancestors[keys[0]]))
        self.regions, self.locations = tuple(regions), locations
        # Defensive verification of the contracted graph before any model executes.
        if regions:
            TaskGraph(Task(r.id, lambda _: None, r.dependencies) for r in regions)
        self._functions = tuple(self._region_function(region) for region in regions)
        self.native = jax.jit(self._native) if all(m.backend == "jax" for m in graph.modules.values()) else None

    def _evaluate(self, event, get, initial, sequences, parameters):
        name, k = event.key
        module = self.graph.modules[name]
        previous = initial[name] if event.previous is None else get(event.previous)
        inputs = {}
        for port, producer, source_module, source_port, external in event.inputs:
            if external is not None:
                inputs[port] = sequences[external][k-self.start_tick]
            else:
                value = initial[source_module] if producer is None else get(producer)
                inputs[port] = value.outputs[source_port]
        context = ModuleContext(jnp.asarray(k, jnp.int32), k*self.dt, self.dt, module.every*self.dt)
        with jax.named_scope(name):
            following = module.step(previous.state, inputs, parameters[name], context)
        check_outputs(module, following, previous)
        return following

    def _region_function(self, region):
        def execute(boundaries, initial, sequences, parameters):
            values = {}
            def get(key):
                if key in values:
                    return values[key]
                producer, index = self.locations[key]
                return boundaries[producer][index]
            for key in region.events:
                values[key] = self._evaluate(self._events[key], get, initial, sequences, parameters)
            return tuple(values[key] for key in region.events)
        return jax.jit(execute) if region.backend == "jax" else execute

    def _assemble(self, get, state):
        end = self.start_tick + self.ticks
        final, traces, updated = {}, {}, {}
        for name, module in self.graph.modules.items():
            last = self._source_event(name, end-1)
            final[name] = state.modules[name] if last is None else get(last)
            values = []
            for k in range(self.start_tick, end):
                key = self._source_event(name, k)
                values.append((state.modules[name] if key is None else get(key)).outputs)
            traces[name] = jax.tree.map(lambda *xs: jnp.stack(xs), *values)
            updated[name] = jnp.arange(self.start_tick, end) % module.every == 0
        return ModuleGraphState(jnp.asarray(end, dtype=state.tick.dtype), final), ModuleGraphTrace(
            jnp.arange(self.start_tick, end, dtype=state.tick.dtype), traces, updated)

    def _native(self, state, sequences, parameters):
        """Pure all-JAX scan: long horizons need not unroll the numerical graph."""
        tick = self.graph.make_tick(self.dt)
        def body(current, offset):
            inputs = {name: values[offset] for name, values in sequences.items()}
            return tick(current, inputs, parameters)
        return jax.lax.scan(body, state, jnp.arange(self.ticks))

    def run(self, state, sequences, parameters, *, max_workers=2, on_complete=None, probe=None, phase="execute"):
        if np.shape(state.tick) != () or not np.issubdtype(state.tick.dtype, np.integer):
            raise ValueError("graph tick must be an integer scalar")
        if int(state.tick) != self.start_tick:
            raise ValueError("state tick must match the compiled start_tick")
        self.graph.validate_values(state, sequences, parameters, self.ticks)
        tasks = []
        for region, function in zip(self.regions, self._functions, strict=True):
            def run(dependencies, function=function, region=region):
                args = (dict(dependencies), state.modules, sequences, parameters)
                if probe is None:
                    return function(*args)
                names = sorted({name for name, tick in region.events})
                label = names[0] if len(region.events) == 1 else f"region/{region.id}"
                return probe.call(label, function, *args, phase=phase,
                                  metadata=dict(backend=region.backend, events=region.events,
                                                modules=names, scope="module" if len(region.events) == 1 else "region",
                                                compilation="included on cache miss"))
            tasks.append(Task(region.id, run, region.dependencies))
        outputs = DependencyExecutor(max_workers=max_workers).run(
            TaskGraph(tasks), on_complete=on_complete) if tasks else {}
        def get(key):
            producer, index = self.locations[key]
            return outputs[producer][index]
        return jax.block_until_ready(self._assemble(get, state))

    def describe(self):
        return [{"id": r.id, "backend": r.backend, "events": list(r.events),
                 "dependencies": list(r.dependencies), "host_ancestors": sorted(r.host_ancestors)}
                for r in self.regions]
