"""Host boundary consumed by CoSimulationRunner; MATLAB transport is separate."""
from dataclasses import dataclass
from typing import Mapping, Protocol
import numpy as np


@dataclass(frozen=True)
class Port:
    name: str
    unit: str
    shape: tuple[int, ...]
    frame: str
    quantity: str = ""
    reference_point: str = ""
    dtype: str = "float64"


@dataclass(frozen=True)
class Capabilities:
    resettable: bool
    snapshot: bool
    variable_communication_step: bool
    direct_feedthrough: bool


@dataclass(frozen=True)
class ExternalSample:
    time_s: float
    values: Mapping[str, np.ndarray]


class ExternalComponent(Protocol):
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    capabilities: Capabilities

    def initialize(self, start_time_s: float, parameters: Mapping[str, object]) -> ExternalSample: ...
    def advance_to(self, target_time_s: float, held_inputs: Mapping[str, np.ndarray]) -> ExternalSample:
        """Advance from current time with ZOH inputs; return actual output time."""
        ...
    def reset(self) -> ExternalSample: ...
    def close(self) -> None: ...
