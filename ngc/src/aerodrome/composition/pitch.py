"""First assembly: reuse the readable pitch loop, do not invent a generic F16 state."""
from dataclasses import dataclass
from typing import NamedTuple
from aerodrome.core.signals import PitchState, GaussianState
from aerodrome.systems.pitch_loop import Blocks, initialize, make_step
from .contracts import SystemSpec


class PitchInitial(NamedTuple):
    truth: PitchState
    navigation_prior: GaussianState


@dataclass(frozen=True)
class PitchAssembly:
    blocks: Blocks = Blocks()
    backend: str = "jax"
    initial_signals: tuple[str, ...] = ("truth", "prior", "goal", "held_reference", "held_command")
    systems: tuple[SystemSpec, ...] = (
        SystemSpec("sensor", "discrete", ("truth",), ("measurement",), equation="sensor.pitch"),
        SystemSpec("navigation_correct", "discrete", ("prior", "measurement"), ("navigation",),
                   after=("sensor",), equation="kalman.correct"),
        SystemSpec("guidance", "discrete", ("navigation", "goal", "held_reference"), ("reference",),
                   ("reference",), ("navigation_correct",), "guidance.slew"),
        SystemSpec("control", "discrete", ("navigation", "reference", "held_command"), ("command",),
                   ("controller", "command"), ("guidance",), "control.pid"),
        SystemSpec("actuator", "algebraic", ("command",), ("actuator",), equation="actuator.ideal"),
        SystemSpec("physics", "continuous", ("truth", "actuator"), ("next_truth",),
                   ("truth",), ("actuator",), "pitch.linear / integration.rk4"),
        SystemSpec("navigation_predict", "discrete", ("navigation", "actuator"), ("next_prior",),
                   ("navigation_prior",), ("actuator",), "kalman.predict"),
    )

    def initialize(self, initial_conditions, key):
        return initialize(initial_conditions.truth, initial_conditions.navigation_prior, key)

    def make_tick(self, schedule):
        step = make_step(schedule, self.blocks)

        def tick(state, goal, parameters, context):
            # Compatibility with the original LoopState. World owns the clock.
            return step(state._replace(tick=context.tick), goal, parameters)

        return tick
