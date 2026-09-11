"""Wrap a pure subsystem as a host-owned co-simulation partition."""
from aerodrome.adapters.external import Capabilities, ExternalSample


class FunctionalComponent:
    capabilities = Capabilities(True, False, True, False)

    def __init__(self, inputs, outputs, initialize, transition):
        self.inputs, self.outputs = tuple(inputs), tuple(outputs)
        self._initialize, self._transition = initialize, transition
        self._ready = False

    def initialize(self, start_time_s, parameters):
        self._start, self._parameters = start_time_s, parameters
        self._state, values = self._initialize(parameters)
        self._time, self._ready = start_time_s, True
        return ExternalSample(self._time, values)

    def advance_to(self, target_time_s, held_inputs):
        if not self._ready or target_time_s <= self._time:
            raise RuntimeError("component must be initialized and time must advance")
        self._state, values = self._transition(
            self._state, held_inputs, self._parameters, target_time_s - self._time)
        self._time = target_time_s
        return ExternalSample(self._time, values)

    def reset(self):
        if not self._ready:
            raise RuntimeError("component must be initialized before reset")
        return self.initialize(self._start, self._parameters)

    def close(self):
        self._ready = False
