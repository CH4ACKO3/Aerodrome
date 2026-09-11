"""Inspectable contracts. The assembly still owns its explicit equations/wiring."""
from dataclasses import dataclass
from typing import Callable, Protocol
from aerodrome.core.clock import Schedule


@dataclass(frozen=True)
class SystemSpec:
    name: str
    kind: str  # algebraic, continuous, discrete
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    state_slots: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    equation: str = ""


def validate_systems(systems, initial_signals):
    """Validate an explicitly ordered plan; never silently reorder equations."""
    names, signals, owners, writers = set(), set(initial_signals), set(), set()
    for system in systems:
        if not system.name or system.name in names:
            raise ValueError(f"duplicate or empty system name: {system.name}")
        if system.kind not in {"algebraic", "continuous", "discrete"}:
            raise ValueError(f"unsupported system kind: {system.kind}")
        if set(system.after) - names:
            raise ValueError(f"{system.name}: dependency is not scheduled earlier")
        if set(system.reads) - signals:
            raise ValueError(f"{system.name}: missing input or unsupported algebraic loop")
        if set(system.state_slots) & owners:
            raise ValueError(f"{system.name}: state has multiple update owners")
        if set(system.writes) & writers:
            raise ValueError(f"{system.name}: signal has multiple writers")
        names.add(system.name)
        owners.update(system.state_slots)
        writers.update(system.writes)
        signals.update(system.writes)


class Assembly(Protocol):
    backend: str
    systems: tuple[SystemSpec, ...]
    initial_signals: tuple[str, ...]

    def initialize(self, initial_conditions, key): ...
    def make_tick(self, schedule: Schedule) -> Callable:
        """Return (entity_state, input, parameters, context) -> (state, record)."""
        ...
