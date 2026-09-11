import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from aerodrome.rendering import (Scene,RenderEntity,Frame,Pose,RendererSession,HeadlessBackend,
    RenderConfig,Camera,JointValue,RenderReceipt,Capabilities,MessageBackend,
    THREE_Y_UP,UNREAL_Z_UP_CM,EngineCoordinates)


def scene():
    return Scene((RenderEntity("plane"),))


def frame():
    return Frame(0,0,0.,0,(0,0),(Pose((1.,2.,-3.),(1.,0,0,0)),))


def test_lifecycle_bounded_submit_seek_and_no_fake_fps():
    backend = HeadlessBackend()
    with RendererSession(backend,scene()) as session:
        assert session.ready
        camera = Camera((0.,0.,-10.),(0.,0.,0.),up_ned=(1.,0.,0.))
        assert session.submit(frame(),camera=camera,joints=(JointValue("plane","elevator",.1),))==1
        assert session.submit(frame()) is None
        assert session.poll()[0].status=="completed"
        assert session.completed==1 and session.meter.snapshot()["fps"]==0
        assert session.submit(frame())==2  # repeat/seek frame allowed, transport ID advances
        assert session.poll()[0].sequence==2
    session.close()
    assert backend.last_request is None
    with pytest.raises(RuntimeError):
        session.submit(frame())


def test_capability_failure_and_cleanup():
    class Minimal(HeadlessBackend):
        def open(self,scene,config):
            super().open(scene,config)
            return Capabilities(modes=("live",))
    backend = Minimal()
    with pytest.raises(ValueError,match="mode"):
        RendererSession(backend,scene(),RenderConfig(mode="offline"))
    assert not backend._open
    with RendererSession(Minimal(),scene()) as session:
        with pytest.raises(ValueError,match="articulations"):
            session.submit(frame(),joints=(JointValue("plane","rudder",.1),))
        assert session.ready


@pytest.mark.parametrize("status",["failed","dropped"])
def test_offline_failure_not_silently_dropped(status):
    backend = HeadlessBackend()
    with RendererSession(backend,scene(),RenderConfig(mode="offline")) as session:
        sequence = session.submit(frame())
        backend._receipt = RenderReceipt(sequence,status,"test error")
        with pytest.raises(RuntimeError):
            session.poll()
        assert not session.ready


def test_ack_identity_and_actual_presented_counter():
    backend = HeadlessBackend()
    with RendererSession(backend,scene()) as session:
        sequence = session.submit(frame())
        backend._receipt = RenderReceipt(sequence,"presented")
        session.poll()
        assert session.meter.snapshot()["presented_frames"]==1
        backend._receipt = RenderReceipt(sequence,"presented")
        with pytest.raises(RuntimeError,match="duplicate"):
            session.poll()
        assert session.meter.snapshot()["presented_frames"]==1


def test_thread_affinity():
    with RendererSession(HeadlessBackend(),scene()) as session:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(session.submit,frame())
            with pytest.raises(RuntimeError,match="owner"):
                future.result()


def test_pending_timeout():
    now = [0.]
    with RendererSession(HeadlessBackend(),scene(),RenderConfig(request_timeout_s=1.),clock=lambda:now[0]) as session:
        session.submit(frame())
        now[0]=2.
        with pytest.raises(TimeoutError):
            session.poll()
        assert not session.ready


def test_message_transport_contract():
    class Loopback:
        def __init__(self):
            self.reply = None
            self.closed = False
        def exchange(self,message):
            self.open = json.loads(message)
            return json.dumps(dict(protocol_version=1,kind="ready",capabilities=dict(modes=["live","offline"],camera=True,articulations=True)))
        def send(self,message):
            self.request = json.loads(message)
            self.reply = json.dumps(dict(protocol_version=1,kind="receipt",sequence=self.request["sequence"],status="completed"))
        def receive(self):
            message,self.reply = self.reply,None
            return message
        def close(self):
            self.closed = True
    transport = Loopback()
    with RendererSession(MessageBackend(transport),scene()) as session:
        session.submit(frame(),joints=(JointValue("plane","elevator",.2),))
        session.poll()
        assert transport.open["scene"]["position_unit"]=="m"
        assert transport.request["frame"]["tick"]==0
        assert transport.request["joints"][0]["axis_local"]==[0.,1.,0.]
        assert session.completed==1
    assert transport.closed


def test_engine_axes_units_and_handedness():
    pose = frame().poses[0]
    p,R = THREE_Y_UP.convert_pose(pose)
    np.testing.assert_allclose(p,[2,3,-1])
    np.testing.assert_allclose(R,np.eye(3))
    p,R = UNREAL_Z_UP_CM.convert_pose(pose)
    np.testing.assert_allclose(p,[100,200,300])
    np.testing.assert_allclose(R,np.eye(3))
    # Positive canonical pitch points nose up; UE forward +X rotates toward +Z.
    pose = Pose((0,0,0),(np.sqrt(.5),0,np.sqrt(.5),0))
    _,R = UNREAL_Z_UP_CM.convert_pose(pose)
    np.testing.assert_allclose(R@np.array([1,0,0]),[0,0,1],atol=1e-14)
    assert np.linalg.det(R)==pytest.approx(1.)
    with pytest.raises(ValueError,match="handedness"):
        EngineCoordinates(tuple(map(tuple,np.eye(3))),((1,0,0),(0,1,0),(0,0,-1)))


def test_camera_joint_validation():
    with pytest.raises(ValueError):
        Camera((0,0,0),(0,0,0))
    with pytest.raises(ValueError):
        JointValue("plane","rudder",.1,axis_local=(0,0,0))
    with RendererSession(HeadlessBackend(),scene()) as session:
        with pytest.raises(ValueError):
            session.submit(frame(),joints=(JointValue("missing","rudder",.1),))
