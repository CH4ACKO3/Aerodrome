"""Scenario and evaluation are experiment concerns, not ECS systems."""
from dataclasses import dataclass
from typing import NamedTuple
import jax


@dataclass(frozen=True)
class Scenario:
    id: str
    seed: int
    initial_conditions: object
    inputs: object
    parameters: object
    steps: int

    def __post_init__(self):
        if not self.id or type(self.steps) is not int or self.steps < 1:
            raise ValueError("scenario needs an ID and a positive decision-step count")


class Evaluation(NamedTuple):
    final_state: object
    trace: object
    metrics: dict


def evaluate(world, scenario, metrics, *, compile=True):
    state = world.reset(scenario.seed, scenario.initial_conditions)
    inputs = world.pack(scenario.inputs)
    world.validate(state, inputs, scenario.parameters)
    run = lambda s, u, p: world.rollout(s, u, p, steps=scenario.steps)
    if compile:
        run = jax.jit(run)
    final, trace = run(state, inputs, scenario.parameters)
    return Evaluation(final, trace, {name: fn(final, trace) for name, fn in metrics.items()})
