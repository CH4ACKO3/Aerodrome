from threading import Event
import jax
import numpy as np
import pytest
from aerodrome.runners.dataflow import Task, TaskGraph, DependencyExecutor, GraphExecutionError
from aerodrome.runners.graph_world import GraphWorldRunner
from test_world import setup, close


def test_cross_tick_progress_without_global_barrier():
    # No sleeps or speed assumptions: slow@0 waits until fast@1 has actually run.
    fast_second = Event()
    slow_started = Event()

    def slow(_):
        slow_started.set()
        assert fast_second.wait(5), "global barrier prevented independent next tick"
        return 10

    def fast_first(_):
        assert slow_started.wait(5)
        return 1

    def fast_next(d):
        fast_second.set()
        return d[("fast", 0)] + 1

    graph = TaskGraph((Task(("slow", 0), slow), Task(("fast", 0), fast_first),
                       Task(("fast", 1), fast_next, (("fast", 0),))))
    result = DependencyExecutor(max_workers=2).run(graph)
    assert result[("fast", 1)] == 2 and result[("slow", 0)] == 10


def test_within_tick_branches_and_cross_tick_feedback():
    a_started, b_started = Event(), Event()
    def aero(_):
        a_started.set()
        assert b_started.wait(5)
        return 3
    def engine(_):
        b_started.set()
        assert a_started.wait(5)
        return 4
    graph = TaskGraph((Task("aero@0", aero), Task("engine@0", engine),
                       Task("forces@0", lambda d: sum(d.values()), ("aero@0", "engine@0")),
                       Task("state@1", lambda d: d["forces@0"] * 0.1, ("forces@0",))))
    result = DependencyExecutor(max_workers=2).run(graph)
    np.testing.assert_allclose(result["state@1"], 0.7)


@pytest.mark.parametrize("chunk_ticks", [1, 5, 64])
def test_graph_world_matches_monolithic_rollout_from_nonzero_time(case, chunk_ticks):
    world, state, inputs, parameters = setup(case, ("aircraft", "target"))
    state, _ = world.step(state, inputs, parameters)
    expected = jax.jit(lambda s, p: world.rollout(s, inputs, p, steps=3))(state, parameters)
    runner = GraphWorldRunner(world, max_workers=2, chunk_ticks=chunk_ticks)
    plan = runner.plan(state, inputs, parameters, steps=3)
    for key, task in plan.graph.tasks.items():
        assert all(parent[0] == key[0] for parent in task.depends_on)
    actual = runner.run(state, inputs, parameters, steps=3)
    close(actual, expected)
    assert int(actual[0].tick) == 16


def test_graph_failure_stops_descendants_and_drains_other_workers():
    started, release, finished = Event(), Event(), Event()
    def fail(_):
        assert started.wait(5)
        release.set()
        raise ValueError("bad model output")
    def independent(_):
        started.set()
        assert release.wait(5)
        finished.set()
    graph = TaskGraph((Task("bad", fail),
                       Task("independent", independent),
                       Task("child", lambda _: pytest.fail("failed dependency must not run"), ("bad",))))
    with pytest.raises(GraphExecutionError, match="bad") as error:
        DependencyExecutor(max_workers=2).run(graph)
    assert isinstance(error.value.__cause__, ValueError)
    assert finished.is_set()  # no background mutation remains after the error


def test_reject_invalid_graph_before_executing_tasks():
    with pytest.raises(ValueError, match="cycle"):
        TaskGraph((Task("a", lambda _: 0, ("b",)), Task("b", lambda _: 0, ("a",))))
    with pytest.raises(ValueError, match="unknown"):
        TaskGraph((Task("a", lambda _: 0, ("missing",)),))
    with pytest.raises(ValueError, match="unique"):
        TaskGraph((Task("a", lambda _: 0), Task("a", lambda _: 1)))
