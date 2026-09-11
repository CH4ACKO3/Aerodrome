import json
from concurrent.futures import ThreadPoolExecutor
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aerodrome.adapters.external import Port
from aerodrome.composition.module_graph import ModuleGraph, ModuleGraphState, InputBinding
from aerodrome.models.sampled_sensor import SampledSensor, SensorParameters
from aerodrome.runners.module_compiler import CompiledModuleGraph
from aerodrome.telemetry import Channel, Recorder, ChunkWriter, PerformanceProbe


def sensor_case(every=2, delay=1, dtype="float64"):
    port = Port("truth", "rad", (), "body", quantity="angle", dtype=dtype)
    sensor = SampledSensor("angle_sensor", port, every, delay)
    graph = ModuleGraph((sensor.module(),), inputs=(port,), bindings=(InputBinding("truth", (sensor.id, "truth")),))
    initial = ModuleGraphState(jnp.asarray(0, jnp.int32), {sensor.id: sensor.initialize(jax.random.key(42))})
    p = SensorParameters(*[jnp.asarray(v, dtype) for v in (0., 0., 0., 0., 0., -100., 100.)])
    p.validate()
    return sensor, graph, initial, {sensor.id: p}


def test_sensor_delay_sampling_hold_and_no_duplicate_freshness():
    sensor, graph, state, p = sensor_case()
    inputs = {"truth": jnp.arange(6., dtype=jnp.float64)}
    final, trace = CompiledModuleGraph(graph, ticks=6, physics_dt_s=.1).native(state, inputs, p)
    out = trace.outputs[sensor.id]
    np.testing.assert_array_equal(out["value"], [0, 0, 0, 2, 2, 4])
    np.testing.assert_array_equal(out["fresh"], [False, True, False, True, False, True])
    np.testing.assert_array_equal(out["sample_tick"], [-1, 0, 0, 2, 2, 4])
    np.testing.assert_array_equal(out["valid"], [False, True, True, True, True, True])
    assert int(final.tick) == 6


def test_sensor_quantization_bounds_dropout_and_float32():
    sensor, graph, state, p = sensor_case(1, 0, "float32")
    p[sensor.id] = p[sensor.id]._replace(bias=jnp.float32(.1), resolution=jnp.float32(.5), upper=jnp.float32(1.))
    run = CompiledModuleGraph(graph, ticks=3, physics_dt_s=.1).native
    inputs = {"truth": jnp.array([.23, .7, 3.], jnp.float32)}
    _, trace = run(state, inputs, p)
    np.testing.assert_allclose(trace.outputs[sensor.id]["value"], [.5, 1, 1])
    assert trace.outputs[sensor.id]["value"].dtype == jnp.float32
    p[sensor.id] = p[sensor.id]._replace(dropout_probability=jnp.float32(1))
    _, trace = run(state, inputs, p)
    assert not np.any(trace.outputs[sensor.id]["valid"])


def test_sensor_randomness_matches_chunking_and_batch():
    sensor, graph, state, p = sensor_case(2, 3)
    p[sensor.id] = p[sensor.id]._replace(noise_std=jnp.asarray(.1), drift_std_per_sqrt_s=jnp.asarray(.02), dropout_probability=jnp.asarray(.3))
    inputs = {"truth": jnp.arange(8.)}
    full = CompiledModuleGraph(graph, ticks=8, physics_dt_s=.1).native
    expected, trace = full(state, inputs, p)
    middle, first = CompiledModuleGraph(graph, ticks=3, physics_dt_s=.1).native(state, {"truth": inputs["truth"][:3]}, p)
    final, second = CompiledModuleGraph(graph, ticks=5, start_tick=3, physics_dt_s=.1).native(middle, {"truth": inputs["truth"][3:]}, p)
    for a, b in zip(jax.tree.leaves((expected, trace)), jax.tree.leaves((final, jax.tree.map(lambda a,b: jnp.concatenate((a,b)), first, second))), strict=True):
        np.testing.assert_allclose(a, b)
    states = jax.tree.map(lambda x: jnp.stack((x,x)), state)
    batch = jax.jit(jax.vmap(full, in_axes=(0,None,None)))(states, inputs, p)
    for a, b in zip(jax.tree.leaves((expected, trace)), jax.tree.leaves(batch), strict=True):
        np.testing.assert_allclose(a, b[0])


def test_sensor_parameter_validation():
    _, _, _, params = sensor_case()
    p = params["angle_sensor"]
    for invalid in (p._replace(noise_std=-1), p._replace(dropout_probability=2), p._replace(lower=200)):
        with pytest.raises(ValueError):
            invalid.validate()


def test_vector_sensor_namespaces_and_delayed_packet_loss():
    from aerodrome.composition.module_graph import ModuleContext
    port = Port("truth", "rad/s", (3,), "body", quantity="angular_rate")
    sensor = SampledSensor("gyro", port, delay_ticks=1)
    other = SampledSensor("other_gyro", port, delay_ticks=1)
    state = sensor.initialize(jax.random.key(1))
    assert not np.array_equal(state.state.key_data, other.initialize(jax.random.key(1)).state.key_data)
    p = SensorParameters(jnp.zeros(3), jnp.array([.1,.2,.3]), 0., 0., 0., -100.,100.)
    p.validate((3,))
    def advance(value,k,p):
        return sensor.step(value.state,{"truth":jnp.ones(3)*k},p,ModuleContext(jnp.int32(k),k*.1,.1,.1))
    first = jax.jit(advance)(state,0,p)
    second = jax.jit(advance)(first,1,p._replace(dropout_probability=1.))
    np.testing.assert_allclose(second.outputs["value"],[.1,.2,.3])
    assert bool(second.outputs["fresh"])
    third = jax.jit(advance)(second,2,p)
    assert not bool(third.outputs["fresh"]) and bool(third.outputs["valid"])
    np.testing.assert_array_equal(third.outputs["value"],second.outputs["value"])


def test_recorder_projects_jitted_batch_and_roundtrips_chunks(tmp_path):
    from batch_rollout import build_batch
    from aerodrome.core.signals import PitchGoal
    batch, state, params, _ = build_batch(2)
    recorder = Recorder((Channel("pitch", lambda s,t: s.entities[0].truth.pitch_rad, "rad"),
                         Channel("tick", lambda s,t: s.tick, "tick")))
    _, values = jax.jit(lambda s: batch.rollout_constant(s, (PitchGoal(jnp.ones(2)*.1),), params, steps=3, record=recorder))(state)
    writer = ChunkWriter(tmp_path, recorder, metadata={"seed":42})
    path = writer.write(0, values)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["pitch"], values["pitch"])
    manifest = json.loads(path.with_suffix(".json").read_text())
    assert manifest["channels"]["pitch"]["unit"] == "rad"
    assert manifest["axes"] == ["time", "world"]
    with pytest.raises(FileExistsError):
        writer.write(0, values)
    with pytest.raises(ValueError):
        writer.write(1, {"pitch":values["pitch"], "tick":values["tick"][:1]})
    writer.write(1, jax.tree.map(lambda x:x[:1], values))


def test_probe_thread_safety_errors_and_trace_export(tmp_path):
    probe = PerformanceProbe()
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda i:probe.call("task", lambda:jnp.asarray(i)+1), range(12)))
    assert len(results) == 12 and len(probe.snapshot()) == 12
    def fail():
        raise RuntimeError("intentional")
    with pytest.raises(RuntimeError):
        probe.call("bad", fail)
    assert probe.snapshot()[-1]["status"] == "error"
    probe.write(tmp_path/"profile.json")
    assert len(json.loads((tmp_path/"profile.trace.json").read_text())["traceEvents"]) == 13


@pytest.mark.parametrize("fuse", [False, True])
def test_module_probe_counts_actual_updates_and_preserves_results(fuse):
    from compiled_modules import build_case
    graph, state, inputs, p, _ = build_case(ticks=4)
    program = CompiledModuleGraph(graph, ticks=4, physics_dt_s=.02, fuse=fuse)
    probe = PerformanceProbe()
    expected = program.run(state, inputs, p)
    actual = program.run(state, inputs, p, probe=probe, phase="warm")
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual),strict=True):
        np.testing.assert_allclose(a,b)
    events = probe.snapshot()
    assert len(events) == len(program.regions)
    assert all(e["duration_ns"] >= 0 and e["phase"] == "warm" for e in events)
    if not fuse:
        assert sum(e["name"] == "engine" for e in events) == 1
        assert sum(e["name"] == "body" for e in events) == 4
        assert all(e["metadata"]["scope"] == "module" for e in events)


def test_host_module_probe_does_not_repeat_external_calls():
    from compiled_modules import build_case
    graph,state,inputs,p,calls = build_case(backend="host",ticks=8)
    program = CompiledModuleGraph(graph,ticks=8,physics_dt_s=.02)
    probe = PerformanceProbe()
    program.run(state,inputs,p,probe=probe)
    assert calls == [0,4]
    host = [e for e in probe.snapshot() if e["metadata"]["backend"] == "host"]
    assert len(host) == 2 and all(e["name"] == "engine" for e in host)
