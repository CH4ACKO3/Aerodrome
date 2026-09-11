"""Actual World -> render projection -> archive/live interface -> 60 FPS replay.

This example has no graphics backend: presented FPS stays zero intentionally.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import jax
import jax.numpy as jnp
from aerodrome.models.rigid_body import RigidBody6DoF, mass_properties, BodyLoads
from aerodrome.composition import build_world,WorldSpec,EntitySpec
from aerodrome.composition.rigid_body import RigidBodyAssembly
from aerodrome.core.clock import Schedule
from aerodrome.rendering import (Scene,RenderEntity,rigid_body_pose,make_projection,
    snapshot,TrajectoryWriter,read_trajectory,resample,LatestFrameStream,RateMeter)


def main():
    jax.config.update("jax_enable_x64",True)
    folder = Path("artifacts/rendering_data")/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    model = RigidBody6DoF()
    world = build_world(WorldSpec((EntitySpec("aircraft",RigidBodyAssembly(model)),),
                                 Schedule(physics_dt_s=.01),ticks_per_step=10))
    state = world.reset(0,{"aircraft":model.initialize(position_ned_m=[0,0,-1000],
                         velocity_body_m_s=[100,0,0],omega_body_rad_s=[0,0,.05])})
    params = world.parameters({"aircraft":mass_properties(1000.,jnp.eye(3)*1000, [0,0,0])})
    inputs = (BodyLoads(jnp.zeros(3),jnp.zeros(3)),)
    scene = Scene((RenderEntity("aircraft"),))
    project = make_projection(.01,(lambda s:rigid_body_pose(s.entities[0]),))
    def chunk(s,p):
        def tick(x,_):
            y,_ = world.tick(x,inputs,p)
            return y,project(y)
        return jax.lax.scan(tick,s,None,length=10)
    compiled = jax.jit(chunk)
    jax.block_until_ready(compiled(state,params))  # warmup is not measured
    meter,stream = RateMeter(),LatestFrameStream()
    with TrajectoryWriter(folder/"physics.jsonl",scene) as writer:
        writer.append(snapshot(scene,project(state)))
        for _ in range(100):
            state,samples = compiled(state,params)
            meter.simulation_completed(state,ticks=10,physics_dt_s=.01)
            # Batched transfer once per chunk; archive all 10 projected ticks.
            host = jax.device_get(samples)
            frames = [snapshot(scene,jax.tree.map(lambda x:x[i],host)) for i in range(10)]
            for frame in frames:
                writer.append(frame)
            stream.publish(frames[-1])
        live = stream.take()  # deliberately slow consumer, tests newest-wins path
    measured = meter.snapshot()
    loaded_scene,frames = read_trajectory(folder/"physics.jsonl")
    count = 0
    with TrajectoryWriter(folder/"replay_60fps.jsonl",loaded_scene) as writer:
        for frame in resample(frames,60):
            writer.append(frame)
            count += 1
    assert len(frames)==1001 and count==601 and live.frame.tick==1000
    summary = dict(physics_dt_s=.01,nominal_simulation_tps=100,target_replay_fps=60,
                   physics_snapshots=len(frames),replay_frames=count,live_last_tick=live.frame.tick,
                   live=stream.statistics(),measured=measured,
                   timing_scope="compute + projected host transfers + archive writing; excludes warmup and offline resampling",
                   renderer="none; measured FPS=0 because no presentation occurred")
    stream.close()
    (folder/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(str(folder.resolve()))
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
