"""Experiment DAG, deliberately separate from the feedback systems in a world."""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Stage:
    id: str
    run: object  # callable receiving only its declared dependency outputs
    depends_on: tuple[str, ...] = ()


class Pipeline:
    def __init__(self, stages):
        stages = tuple(stages)
        by_id = {s.id: s for s in stages}
        if len(by_id) != len(stages) or any(not s.id or not callable(s.run) for s in stages):
            raise ValueError("stage IDs must be unique/nonempty and run must be callable")
        if any(set(s.depends_on) - by_id.keys() for s in stages):
            raise ValueError("unknown pipeline dependency")
        order, pending = [], dict(by_id)
        while pending:
            ready = [s for s in pending.values() if set(s.depends_on) <= {x.id for x in order}]
            if not ready:
                raise ValueError("pipeline dependency cycle")
            for stage in ready:
                order.append(stage)
                del pending[stage.id]
        self.stages = tuple(order)

    def run(self):
        outputs = {}
        for stage in self.stages:
            outputs[stage.id] = stage.run(MappingProxyType({key: outputs[key] for key in stage.depends_on}))
        return MappingProxyType(outputs)
