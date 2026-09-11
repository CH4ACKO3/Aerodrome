"""Synchronized wall-clock spans; never interpreted as exclusive GPU kernel time."""
from collections import defaultdict
from threading import Lock, get_ident
from time import perf_counter_ns
import json
from pathlib import Path
import jax


class PerformanceProbe:
    def __init__(self):
        self._lock = Lock()
        self._events = []

    def call(self, name, function, *args, phase="execute", metadata=None, **kwargs):
        # Drain input production outside the span; synchronize output inside it.
        jax.block_until_ready((args, kwargs))
        start = perf_counter_ns()
        status = "ok"
        try:
            return jax.block_until_ready(function(*args, **kwargs))
        except BaseException:
            status = "error"
            raise
        finally:
            event = dict(name=str(name), phase=phase, start_ns=start,
                         duration_ns=perf_counter_ns()-start, thread_id=get_ident(),
                         status=status, metadata=dict(metadata or {}))
            with self._lock:
                self._events.append(event)

    def snapshot(self):
        with self._lock:
            return [dict(e, metadata=dict(e["metadata"])) for e in self._events]

    def summary(self):
        groups = defaultdict(list)
        for event in self.snapshot():
            groups[(event["name"], event["phase"], event["status"])].append(event["duration_ns"])
        return [dict(name=name, phase=phase, status=status, count=len(values),
                     total_ns=sum(values), mean_ns=sum(values)/len(values), min_ns=min(values), max_ns=max(values))
                for (name, phase, status), values in sorted(groups.items())]

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        events = self.snapshot()
        path.write_text(json.dumps({"clock": "perf_counter_ns", "timing": "synchronized host wall time",
                                    "events": events, "summary": self.summary()}, indent=2), encoding="utf-8")
        chrome = [{"name": e["name"], "cat": e["phase"], "ph": "X", "pid": 1,
                   "tid": e["thread_id"], "ts": e["start_ns"]/1000, "dur": e["duration_ns"]/1000,
                   "args": dict(e["metadata"], status=e["status"])} for e in events]
        path.with_suffix(".trace.json").write_text(json.dumps({"traceEvents": chrome}), encoding="utf-8")
