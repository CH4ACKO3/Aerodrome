"""Fixed-step Jacobi/ZOH co-simulation. No rollback, algebraic solve, or JAX transform."""
from dataclasses import dataclass
import math
from types import MappingProxyType
import numpy as np
from aerodrome.adapters.external import ExternalSample


@dataclass(frozen=True)
class Connection:
    source: tuple[str, str]  # (partition ID, output port)
    target: tuple[str, str]  # (partition ID, input port)


class CoSimulationRunner:
    """Own sessions in a context manager. Feedback uses previous boundary outputs.

    Initial outputs must be functions of initial state/configuration only.
    Direct-feedthrough partitions are rejected; explicitly latch or repartition them.
    """

    def __init__(self, components, connections, *, communication_dt_s, external_inputs=()):
        if not math.isfinite(communication_dt_s) or communication_dt_s <= 0:
            raise ValueError("communication_dt_s must be finite and positive")
        if not components or any(not isinstance(k, str) or not k for k in components):
            raise ValueError("partitions require nonempty string IDs")
        if len({id(c) for c in components.values()}) != len(components):
            raise ValueError("each partition needs an independent component instance")
        self.components = dict(components)
        self.dt = communication_dt_s
        self._inputs, self._outputs = {}, {}
        for name, component in self.components.items():
            if component.capabilities.direct_feedthrough:
                raise ValueError(f"{name}: direct feedthrough requires explicit latching/repartitioning")
            for ports, index in ((component.inputs, self._inputs), (component.outputs, self._outputs)):
                for port in ports:
                    endpoint = (name, port.name)
                    if endpoint in index or not port.name:
                        raise ValueError(f"duplicate or empty port: {endpoint}")
                    index[endpoint] = port
        self._links = {}
        for connection in connections:
            source, target = connection.source, connection.target
            if source not in self._outputs or target not in self._inputs:
                raise ValueError(f"unknown connection: {connection}")
            if target in self._links:
                raise ValueError(f"multiple writers: {target}")
            a, b = self._outputs[source], self._inputs[target]
            if any(getattr(a, field) != getattr(b, field) for field in
                   ("shape", "unit", "frame", "quantity", "reference_point", "dtype")):
                raise ValueError(f"incompatible ports: {connection}; use an explicit conversion adapter")
            self._links[target] = source
        external_inputs = tuple(external_inputs)
        self.external_inputs = frozenset(external_inputs)
        if len(external_inputs) != len(self.external_inputs):
            raise ValueError("duplicate external input")
        if (self.external_inputs & self._links.keys() or
                self.external_inputs | self._links.keys() != self._inputs.keys()):
            raise ValueError("every input needs exactly one connection or external source")
        self._ready, self._failed, self._closed = False, False, False

    @staticmethod
    def _value(value, port):
        array = np.asarray(value)
        if array.shape != port.shape or array.dtype != np.dtype(port.dtype):
            raise ValueError(f"{port.name}: wrong shape/dtype, expected {port.shape}/{port.dtype}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{port.name}: nonfinite value")
        return array.copy()

    def _sample(self, name, sample, expected_time):
        if not math.isfinite(sample.time_s) or not math.isclose(
                sample.time_s, expected_time, rel_tol=0., abs_tol=1e-10):
            raise ValueError(f"{name}: stopped at {sample.time_s}, expected {expected_time}")
        ports = self.components[name].outputs
        if set(sample.values) != {p.name for p in ports}:
            raise ValueError(f"{name}: output ports do not match contract")
        values = {p.name: self._value(sample.values[p.name], p) for p in ports}
        for value in values.values():
            value.setflags(write=False)
        return ExternalSample(expected_time, MappingProxyType(values))

    def initialize(self, parameters, *, start_time_s=0.):
        if self._closed or self._ready:
            raise RuntimeError("use a new runner or reset an initialized runner")
        if set(parameters) != set(self.components) or not math.isfinite(start_time_s):
            raise ValueError("provide parameters for every partition and a finite start time")
        self._start, self._tick = start_time_s, 0
        try:
            self.samples = {name: self._sample(name, c.initialize(start_time_s, parameters[name]), start_time_s)
                            for name, c in self.components.items()}
        except Exception as error:
            self._failed = True
            try:
                self.close()
            except Exception as cleanup:
                error.add_note(f"cleanup also failed: {cleanup}")
            raise
        self._ready, self._failed = True, False
        return MappingProxyType(self.samples)

    def step(self, external_inputs=None):
        if not self._ready or self._failed or self._closed:
            raise RuntimeError("runner is not ready; a failed step requires reset or close")
        external_inputs = {} if external_inputs is None else external_inputs
        if set(external_inputs) != self.external_inputs:
            raise ValueError("provide exactly the declared external inputs")
        # Snapshot every input before advancing ANY partition: order-independent Jacobi.
        held = {name: {} for name in self.components}
        for endpoint, port in self._inputs.items():
            if endpoint in self._links:
                source = self._links[endpoint]
                value = self.samples[source[0]].values[source[1]]
            else:
                value = external_inputs[endpoint]
            held[endpoint[0]][endpoint[1]] = self._value(value, port)
        target = self._start + (self._tick + 1) * self.dt
        try:
            samples = {name: self._sample(name, c.advance_to(target, held[name]), target)
                       for name, c in self.components.items()}
        except Exception:
            # Some sessions may already have advanced; never continue from partial state.
            self._failed = True
            raise
        self.samples, self._tick = samples, self._tick + 1
        return MappingProxyType(samples)

    def reset(self):
        if self._closed or not self._ready:
            raise RuntimeError("initialize before reset")
        if not all(c.capabilities.resettable for c in self.components.values()):
            raise RuntimeError("all partitions must support reset")
        self._failed = True
        samples = {name: self._sample(name, c.reset(), self._start) for name, c in self.components.items()}
        self.samples, self._tick, self._failed = samples, 0, False
        return MappingProxyType(samples)

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        for c in reversed(tuple(self.components.values())):
            try:
                c.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("partition close failures", errors)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as cleanup:
            if exc is None:
                raise
            exc.add_note(f"cleanup also failed: {cleanup}")
        return False
