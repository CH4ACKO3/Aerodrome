import json
from concurrent.futures import ThreadPoolExecutor
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from aerodrome.rendering import (Scene,RenderEntity,Pose,RenderSample,Frame,snapshot,
    rigid_body_pose,make_projection,TrajectoryWriter,read_trajectory,interpolate,
    resample,LatestFrameStream,RateMeter,RateLimiter)


def frame(tick, *, episode=0,world=0,quaternion=(1.,0,0,0)):
    return Frame(world,episode,tick*.01,tick,(tick,tick),(Pose((float(tick),0,0),quaternion),))


def test_projection_jit_vmap_and_immutable_snapshot():
    from aerodrome.models.rigid_body import RigidBody6DoF
    from aerodrome.composition.world import WorldState
    body = RigidBody6DoF("euler321").initialize(euler_rad=[0,0,np.pi/2])
    project = make_projection(.01,(lambda s:rigid_body_pose(s.entities[0],attitude="euler321"),))
    state = WorldState(jnp.array(5),(body,))
    sample = jax.jit(project)(state)
    scene = Scene((RenderEntity("f16"),))
    result = snapshot(scene,sample,world_id=4,episode_id=7)
    assert result.tick==5 and result.time_s==pytest.approx(.05)
    np.testing.assert_allclose(result.poses[0].quaternion_nb,[np.sqrt(.5),0,0,np.sqrt(.5)],atol=1e-14)
    batch = jax.tree.map(lambda x:jnp.stack((x,x)),state)
    assert jax.jit(jax.vmap(project))(batch).poses[0].position_ned_m.shape==(2,3)
    with pytest.raises(ValueError,match="one world"):
        snapshot(scene,jax.vmap(project)(batch))
    mutable = np.array([1.,2,3])
    frozen = Frame(0,0,0.,0,(0,0),(Pose(mutable,[2,0,0,0]),))
    mutable[0]=100.
    assert frozen.poses[0].position_ned_m==(1.,2.,3.)
    assert json.loads(frozen.to_json())["schema_version"]==1


def test_offline_roundtrip_and_exclusive_creation(tmp_path):
    scene = Scene((RenderEntity("body",asset_uri="models/body.glb"),),(.4,.5,100.))
    path = tmp_path/"trajectory.jsonl"
    with TrajectoryWriter(path,scene) as writer:
        writer.append(frame(0))
        writer.append(frame(10))
        writer.append(frame(0,episode=1))
    recovered,frames = read_trajectory(path)
    assert recovered==scene
    assert frames==(frame(0),frame(10),frame(0,episode=1))
    with pytest.raises(FileExistsError):
        TrajectoryWriter(path,scene)
    with pytest.raises(ValueError,match="episode"):
        tuple(resample(frames,60))
    path.write_text(path.read_text().replace('"position_frame": "local_ned"','"position_frame": "enu"'))
    with pytest.raises(ValueError,match="conventions"):
        read_trajectory(path)


def test_interpolation_shortest_arc_and_frame_grid():
    left,right = frame(0),frame(10,quaternion=(0.,0,0,1.))
    mid = interpolate(left,right,.05)
    assert mid.tick is None and mid.source_ticks==(0,10)
    np.testing.assert_allclose(mid.poses[0].position_ned_m,[5,0,0])
    q = mid.poses[0].quaternion_nb
    np.testing.assert_allclose(Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix(),Rotation.from_euler("z",np.pi/2).as_matrix(),atol=1e-14)
    antipodal = interpolate(left,frame(10,quaternion=(-1.,0,0,0)),.05)
    np.testing.assert_allclose(antipodal.poses[0].quaternion_nb,[1,0,0,0])
    assert len(tuple(resample((left,right),60)))==7
    np.testing.assert_allclose([f.time_s for f in resample((left,right),60)],np.arange(7)/60)
    assert len(tuple(resample((left,right),60,playback_speed=2)))==4
    with pytest.raises(ValueError):
        interpolate(left,right,.2)
    with pytest.raises(ValueError):
        interpolate(left,frame(10,episode=1),.05)


def test_live_latest_drop_reset_close_and_wait():
    stream = LatestFrameStream()
    stream.publish(frame(0))
    stream.publish(frame(1))
    stream.publish(frame(2))
    delivery = stream.take()
    assert delivery.sequence==3 and delivery.frame.tick==2
    assert stream.statistics()["dropped_unread"]==2
    assert stream.take() is None
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(stream.take,timeout_s=2.)
        stream.publish(frame(0,episode=1))
        assert future.result(timeout=3).frame.episode_id==1
    with pytest.raises(ValueError,match="stale"):
        stream.publish(frame(3))
    stream.close()
    assert stream.take() is None
    with pytest.raises(RuntimeError):
        stream.publish(frame(1,episode=1))


def test_tps_fps_real_time_and_batch_counts():
    now = [0.]
    meter = RateMeter(clock=lambda:now[0])
    meter.simulation_completed(jnp.ones(2),ticks=100,physics_dt_s=.01,world_count=8)
    for _ in range(30):
        meter.frame_presented()
    now[0]=2.
    report = meter.snapshot()
    assert report["tps"]==50 and report["aggregate_tps"]==400
    assert report["fps"]==15 and report["real_time_factor"]==.5
    assert report["simulated_seconds"]==1
    with pytest.raises(ValueError):
        meter.simulation_completed((),ticks=-1,physics_dt_s=.01)


def test_rate_limiter_no_catchup_burst():
    now = [0.]
    limiter = RateLimiter(20,clock=lambda:now[0])
    assert limiter.ready() and not limiter.ready()
    now[0]=.049
    assert not limiter.ready()
    now[0]=.05
    assert limiter.ready()
    now[0]=10.
    assert limiter.ready() and not limiter.ready()
    with pytest.raises(ValueError):
        RateLimiter(0)


def test_invalid_data_rejected():
    with pytest.raises(ValueError):
        Scene((RenderEntity("x"),RenderEntity("x")))
    with pytest.raises(ValueError):
        Frame(0,0,0.,0,(0,0),(Pose([np.nan,0,0],[1,0,0,0]),))
    with pytest.raises(ValueError):
        Frame(0,0,0.,0,(0,0),(Pose([0,0,0],[0,0,0,0]),))
    stream = LatestFrameStream()
    stream.publish(frame(3))
    with pytest.raises(ValueError):
        stream.publish(frame(2))
    with pytest.raises(ValueError):
        stream.publish(frame(4,world=1))
