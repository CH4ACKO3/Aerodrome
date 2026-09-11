"""Sampled sensor tap + batch chunk logging + separate diagnostic module timings."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from aerodrome.composition import EntitySpec, WorldSpec, build_world
from aerodrome.composition.graph_assembly import GraphAssembly
from aerodrome.composition.module_graph import ModuleGraph, Wire, InputBinding
from aerodrome.models.sampled_sensor import SampledSensor, SensorParameters
from aerodrome.runners.batch import BatchedWorld
from aerodrome.runners.module_compiler import CompiledModuleGraph
from aerodrome.telemetry import Channel, Recorder, ChunkWriter, PerformanceProbe
from compiled_modules import build_case, speed


def main():
    jax.config.update("jax_enable_x64", True)
    graph, initial, sequence, params, _ = build_case(ticks=8)
    sensor = SampledSensor("speed_sensor", speed("truth"), every=2, delay_ticks=1)
    # A diagnostic sensor tap; the original teaching controller keeps its ideal observation.
    wires = [s for s in graph.sources.values() if isinstance(s, Wire)]
    bindings = [s for s in graph.sources.values() if isinstance(s, InputBinding)]
    graph = ModuleGraph((*graph.modules.values(), sensor.module()),
                        (*wires, Wire(("body", "next_speed"), (sensor.id, "truth"), delay=1)),
                        inputs=graph.inputs.values(), bindings=bindings)
    params[sensor.id] = SensorParameters(*map(jnp.asarray, (.02, .01, .001, .1, .01, -100., 100.)))
    params[sensor.id].validate()
    initial = initial._replace(modules={**initial.modules, sensor.id:sensor.initialize(jax.random.key(42))})
    world = build_world(WorldSpec((EntitySpec("plane", GraphAssembly(graph)),), ticks_per_step=1))
    # The batch sampler receives a distinct key per stable World ID and episode.
    def sample(key, template):
        return {"plane":{**template["plane"],sensor.id:sensor.initialize(key)}}
    ids = np.arange(8)
    templates = {"plane":initial.modules}
    batch = BatchedWorld(world, initial_sampler=sample)
    state = batch.reset(42, ids, templates)
    p = world.parameters({"plane":params})
    recorder = Recorder(tuple(Channel(name, lambda w,t,name=name:w.entities[0].modules[sensor.id].outputs[name], unit)
                              for name,unit in (("value","m/s"),("fresh","1"),("valid","1"),("sample_tick","tick")))
                        +(Channel("world_tick",lambda w,t:w.tick,"tick"),))
    run = jax.jit(lambda s,p:batch.rollout_constant(s, ({"target":jnp.full(8,12.)},), p, steps=8, record=recorder))
    probe = PerformanceProbe()
    executable = probe.call("batch_rollout", lambda:run.lower(state,p).compile(), phase="compile")
    probe.call("batch_rollout", executable, state,p,phase="warmup")
    folder = Path("artifacts/instrumented_rollout")
    folder.mkdir(parents=True,exist_ok=True)
    # Each invocation writes a fresh run directory; existing chunks are never overwritten.
    from tempfile import mkdtemp
    logs = Path(mkdtemp(prefix="run-",dir=folder))
    writer = ChunkWriter(logs,recorder,metadata={"seed":42,"world_ids":ids.tolist(),"episode_id":0,
                                              "model":"teaching 1D speed; diagnostic sensor tap"})
    for index in range(2):
        state,values = probe.call("batch_rollout",executable,state,p,phase="warm")
        probe.call("write_chunk",writer.write,index,values,phase="io")

    diagnostic = CompiledModuleGraph(graph,ticks=8,physics_dt_s=world.spec.schedule.physics_dt_s,fuse=False)
    diagnostic.run(initial,sequence,params,probe=probe,phase="cold_diagnostic")
    diagnostic.run(initial,sequence,params,probe=probe,phase="warm_diagnostic")
    probe.write(logs/"performance.json")
    summary = dict(batch_size=8,chunks=2,steps_per_chunk=8,device=str(jax.devices()[0]),
                   diagnostic_events=len(diagnostic.events),logs=str(logs),
                   timing="synchronized wall time; per-module diagnostic disables fusion")
    (folder/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
