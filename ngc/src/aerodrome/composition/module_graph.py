"""Executable port graph with explicit current/previous-tick signal semantics."""
from dataclasses import dataclass
from collections import deque
from types import MappingProxyType
from typing import Callable, NamedTuple, Any
import numpy as np
import jax
import jax.numpy as jnp
from aerodrome.adapters.external import Port


@dataclass(frozen=True)
class Module:
    id: str
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    step: Callable
    backend: str = "jax"
    every: int = 1
    equation: str = ""

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        if not self.id or not callable(self.step) or self.backend not in {"jax", "host"}:
            raise ValueError("module requires an ID, step function and jax/host backend")
        if type(self.every) is not int or self.every < 1:
            raise ValueError("module every must be a positive integer")
        for ports in (self.inputs, self.outputs):
            if len({p.name for p in ports}) != len(ports) or any(not p.name for p in ports):
                raise ValueError(f"{self.id}: duplicate or empty port")


@dataclass(frozen=True)
class Wire:
    source: tuple[str, str]
    target: tuple[str, str]
    delay: int = 0  # read source output at k, or held source output at k-1


@dataclass(frozen=True)
class InputBinding:
    source: str  # name in ModuleGraph.inputs
    target: tuple[str, str]


class ModuleValue(NamedTuple):
    state: Any
    outputs: dict


class ModuleGraphState(NamedTuple):
    tick: Any  # next event index; outputs hold the last published samples
    modules: dict[str, ModuleValue]


class ModuleContext(NamedTuple):
    tick: Any
    time_s: Any
    physics_dt_s: Any
    sample_period_s: Any


class ModuleGraphTrace(NamedTuple):
    tick: Any
    outputs: dict
    updated: dict


def compatible(source, target):
    fields = ("unit", "shape", "frame", "quantity", "reference_point", "dtype")
    return all(getattr(source, field) == getattr(target, field) for field in fields)


def check_outputs(module, value, previous):
    if not isinstance(value, ModuleValue) or set(value.outputs) != {p.name for p in module.outputs}:
        raise ValueError(f"{module.id}: step must return ModuleValue with exactly the declared outputs")
    for port in module.outputs:
        output = value.outputs[port.name]
        if np.shape(output) != port.shape or np.dtype(output.dtype) != np.dtype(port.dtype):
            raise ValueError(f"{module.id}.{port.name}: output shape/dtype does not match port")
    signature = lambda x: (jax.tree.structure(x), [(np.shape(a), np.dtype(a.dtype)) for a in jax.tree.leaves(x)])
    if signature(value.state) != signature(previous.state):
        raise ValueError(f"{module.id}: state structure/shape/dtype changed")


class ModuleGraph:
    def __init__(self, modules, wires=(), *, inputs=(), bindings=()):
        modules, wires, inputs, bindings = tuple(modules), tuple(wires), tuple(inputs), tuple(bindings)
        by_id = {m.id: m for m in modules}
        if not modules or len(by_id) != len(modules):
            raise ValueError("module IDs must be unique and graph must be nonempty")
        external = {p.name: p for p in inputs}
        if len(external) != len(inputs) or any(not p.name for p in inputs):
            raise ValueError("duplicate or empty graph input")
        ins = {(m.id, p.name): p for m in modules for p in m.inputs}
        outs = {(m.id, p.name): p for m in modules for p in m.outputs}
        sources, predecessors = {}, {m.id: set() for m in modules}
        for wire in wires:
            if type(wire.delay) is not int or wire.delay not in (0, 1):
                raise ValueError("wire delay must be 0 or 1 tick")
            if wire.source not in outs or wire.target not in ins:
                raise ValueError("unknown wire port")
            if wire.target in sources:
                raise ValueError(f"multiple input writers: {wire.target}")
            if not compatible(outs[wire.source], ins[wire.target]):
                raise ValueError(f"incompatible wire: {wire}; provide an explicit conversion module")
            sources[wire.target] = wire
            if wire.delay == 0:
                predecessors[wire.target[0]].add(wire.source[0])
        for binding in bindings:
            if binding.source not in external or binding.target not in ins:
                raise ValueError("unknown external input binding")
            if binding.target in sources:
                raise ValueError(f"multiple input writers: {binding.target}")
            if not compatible(external[binding.source], ins[binding.target]):
                raise ValueError(f"incompatible input binding: {binding}")
            sources[binding.target] = binding
        if sources.keys() != ins.keys():
            raise ValueError("every module input needs exactly one source")
        children = {key: [] for key in by_id}
        counts = {key: len(deps) for key, deps in predecessors.items()}
        for key, deps in predecessors.items():
            for dep in deps:
                children[dep].append(key)
        ready, order = deque(key for key, n in counts.items() if n == 0), []
        while ready:
            key = ready.popleft()
            order.append(key)
            for child in children[key]:
                counts[child] -= 1
                if counts[child] == 0:
                    ready.append(child)
        if len(order) != len(modules):
            raise ValueError("same-tick algebraic cycle; solve jointly or declare a justified delay")
        self.modules = MappingProxyType(by_id)
        self.inputs = MappingProxyType(external)
        self.sources = MappingProxyType(sources)
        self.order = tuple(order)

    def make_tick(self, physics_dt_s):
        """Pure all-JAX tick, with explicit sample/hold and previous-tick edges."""
        if any(module.backend != "jax" for module in self.modules.values()):
            raise ValueError("host modules require the partitioned module compiler")

        def tick(state, external_inputs, parameters):
            if np.shape(state.tick) != () or not np.issubdtype(state.tick.dtype, np.integer):
                raise ValueError("graph tick must be an integer scalar")
            current = dict(state.modules)
            updated = {}
            for name in self.order:
                module = self.modules[name]
                inputs = {}
                for port in module.inputs:
                    source = self.sources[(name, port.name)]
                    if isinstance(source, InputBinding):
                        inputs[port.name] = external_inputs[source.source]
                    else:
                        values = state.modules if source.delay else current
                        inputs[port.name] = values[source.source[0]].outputs[source.source[1]]
                previous = state.modules[name]
                context = ModuleContext(state.tick, state.tick*physics_dt_s,
                                        physics_dt_s, module.every*physics_dt_s)
                def update():
                    with jax.named_scope(name):
                        following = module.step(previous.state, inputs, parameters[name], context)
                    check_outputs(module, following, previous)
                    return following
                due = state.tick % module.every == 0
                current[name] = jax.lax.cond(due, update, lambda: previous)
                updated[name] = due
            return ModuleGraphState(state.tick+1, current), ModuleGraphTrace(
                state.tick, {name: value.outputs for name, value in current.items()}, updated)
        return tick

    def validate_values(self, state, sequences, parameters, ticks):
        if set(state.modules) != set(self.modules) or set(parameters) != set(self.modules):
            raise ValueError("provide initial state and parameters for every module")
        if set(sequences) != set(self.inputs):
            raise ValueError("provide exactly the declared graph input sequences")
        for name, port in self.inputs.items():
            value = sequences[name]
            if np.shape(value) != (ticks,) + port.shape or np.dtype(value.dtype) != np.dtype(port.dtype):
                raise ValueError(f"{name}: expected input sequence shape {(ticks,) + port.shape}/{port.dtype}")
        for name, value in state.modules.items():
            check_outputs(self.modules[name], value, value)
