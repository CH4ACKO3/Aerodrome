"""Host task DAG: submit ready nodes without barriers between logical ticks.

Tasks must declare every dependency and treat dependency outputs as immutable.
This executor is outside JIT/grad; each numerical task can itself be JIT-compiled.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Hashable
import jax


@dataclass(frozen=True)
class Task:
    id: Hashable
    run: Callable
    depends_on: tuple[Hashable, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        if not callable(self.run) or len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("task needs a callable and unique dependencies")


class TaskGraph:
    def __init__(self, tasks):
        tasks = tuple(tasks)
        by_id = {task.id: task for task in tasks}
        if len(by_id) != len(tasks) or not tasks:
            raise ValueError("task IDs must be unique and graph must be nonempty")
        children = {key: [] for key in by_id}
        for task in tasks:
            for dependency in task.depends_on:
                if dependency not in by_id:
                    raise ValueError(f"{task.id}: unknown dependency {dependency}")
                children[dependency].append(task.id)
        counts = {task.id: len(task.depends_on) for task in tasks}
        ready = deque(key for key, count in counts.items() if count == 0)
        visited = 0
        while ready:
            key = ready.popleft()
            visited += 1
            for child in children[key]:
                counts[child] -= 1
                if counts[child] == 0:
                    ready.append(child)
        if visited != len(tasks):
            raise ValueError("task dependency cycle; feedback needs an explicit previous-time edge")
        self.tasks = MappingProxyType(by_id)
        self.children = MappingProxyType({key: tuple(value) for key, value in children.items()})


class GraphExecutionError(RuntimeError):
    def __init__(self, task_id):
        self.task_id = task_id
        super().__init__(f"task {task_id!r} failed; running tasks drained, no complete graph result")


class DependencyExecutor:
    def __init__(self, *, max_workers=2):
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        self.max_workers = max_workers

    @staticmethod
    def _execute(task, dependencies):
        # JAX dispatch is asynchronous. A host future is complete only when the
        # returned numerical result is actually ready; no unnecessary NumPy copy.
        return jax.block_until_ready(task.run(dependencies))

    def run(self, graph, *, on_complete=None):
        """Block for the whole graph; workers progress independently within it.

        on_complete(task_id, output) runs in the coordinator, in observed completion
        order. It is for progress/logging, never implicit feedback into tasks.
        """
        counts = {key: len(task.depends_on) for key, task in graph.tasks.items()}
        ready = deque(key for key, count in counts.items() if count == 0)
        results, pending = {}, {}
        pool = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="aerodrome-dataflow")
        try:
            while ready or pending:
                while ready and len(pending) < self.max_workers:
                    key = ready.popleft()
                    task = graph.tasks[key]
                    values = MappingProxyType({dep: results[dep] for dep in task.depends_on})
                    pending[pool.submit(self._execute, task, values)] = key
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                completed = []
                # Check all observed failures before scheduling more work.
                for future in done:
                    key = pending.pop(future)
                    try:
                        results[key] = future.result()
                    except Exception as error:
                        raise GraphExecutionError(key) from error
                    completed.append(key)
                for key in completed:
                    if on_complete is not None:
                        on_complete(key, results[key])
                    for child in graph.children[key]:
                        counts[child] -= 1
                        if counts[child] == 0:
                            ready.append(child)
            return MappingProxyType(results)
        finally:
            for future in pending:
                future.cancel()
            # Threads already running cannot be killed safely. External adapters
            # need their own timeouts and session ownership before using this API.
            pool.shutdown(wait=True, cancel_futures=True)
