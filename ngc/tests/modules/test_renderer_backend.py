import json
import numpy as np
import pytest
from aerodrome.rendering import Scene, RenderEntity, Frame, Pose, RendererSession, HeadlessBackend, RenderConfig, Camera, JointValue, MessageBackend, THREE_Y_UP, UNREAL_Z_UP_CM


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


def test_pending_timeout():
    now = [0.]
    with RendererSession(HeadlessBackend(),scene(),RenderConfig(request_timeout_s=1.),clock=lambda:now[0]) as session:
        session.submit(frame())
        now[0]=2.
        with pytest.raises(TimeoutError):
            session.poll()
        assert not session.ready


@pytest.mark.parametrize("status", ["completed", "presented", "failed", "dropped"])
def test_message_transport_renders_or_reports_failure(status):
    class Loopback:
        def __init__(self):
            self.reply = None
            self.closed = False
        def exchange(self,message):
            self.open = json.loads(message)
            return json.dumps(dict(protocol_version=1,kind="ready",capabilities=dict(modes=["live","offline"],camera=True,articulations=True)))
        def send(self,message):
            self.request = json.loads(message)
            self.reply = json.dumps(dict(protocol_version=1,kind="receipt",sequence=self.request["sequence"],status=status))
        def receive(self):
            message,self.reply = self.reply,None
            return message
        def close(self):
            self.closed = True
    transport = Loopback()
    with RendererSession(MessageBackend(transport),scene(),RenderConfig(mode="offline")) as session:
        session.submit(frame(),joints=(JointValue("plane","elevator",.2),))
        if status in ("failed", "dropped"):
            with pytest.raises(RuntimeError):
                session.poll()
            assert session.completed == 0
        else:
            session.poll()
            assert session.completed == 1
            assert session.meter.snapshot()["presented_frames"] == (status == "presented")
        assert transport.open["scene"]["position_unit"]=="m"
        assert transport.request["frame"]["tick"]==0
        assert transport.request["joints"][0]["axis_local"]==[0.,1.,0.]
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
