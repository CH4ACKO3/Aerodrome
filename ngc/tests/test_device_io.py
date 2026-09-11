from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from aerodrome.adapters.external import Port
from aerodrome.adapters.device_io import InputChannel,OutputChannel,input_module
from aerodrome.adapters.input_devices import KeyboardMouse,PygameInput,OpenCVCamera
from aerodrome.composition.module_graph import Module,ModuleGraph,ModuleGraphState,ModuleValue,Wire,InputBinding
from aerodrome.runners.module_compiler import CompiledModuleGraph


def port():
    return Port("value","1",(),"scalar",dtype="float32")


def test_channel_stale_fresh_disconnect_and_copy():
    now = [0.]
    channel = InputChannel((port(),),clock=lambda:now[0])
    assert not channel.read()["io_valid"]
    value = np.asarray(2.,np.float32)
    channel.publish({"value":value})
    value[...] = 10
    sample = channel.read()
    assert sample["value"]==2 and sample["io_fresh"]
    assert not channel.read(previous_sequence=1)["io_fresh"]
    sample["value"][...] = 20
    assert channel.read()["value"]==2
    now[0]=1.
    assert not channel.read(max_age_s=.5)["io_valid"]
    channel.disconnect()
    assert not channel.read(max_age_s=10)["io_valid"]
    channel.close()
    with pytest.raises(RuntimeError):
        channel.publish({"value":np.asarray(1.,np.float32)})


def test_keyboard_short_tap_repeat_focus_and_motion():
    hub = KeyboardMouse(("pitch_up","pitch_down"))
    hub.key("pitch_up",True)
    hub.key("pitch_up",True)
    hub.key("pitch_up",False)
    sample = hub.channel.read()
    assert not sample["keys"][0] and sample["presses"][0]==1 and sample["releases"][0]==1
    hub.key("pitch_down",True)
    hub.button(0,True)
    hub.motion([4,8],[4,8])
    hub.motion([5,10],[1,2])
    hub.wheel([0,1])
    hub.focus(False)
    sample = hub.channel.read()
    assert not np.any(sample["keys"]) and not np.any(sample["buttons"])
    assert sample["releases"][1]==1
    np.testing.assert_allclose(sample["motion_total"],[5,10])


def test_hybrid_graph_source_and_output_sink():
    channel = InputChannel((port(),))
    channel.publish({"value":np.asarray(3.,np.float32)})
    sink = OutputChannel((port(),))
    graph = ModuleGraph((channel.module("input",max_age_s=100),sink.module("output")),
                        (Wire(("input","value"),("output","value")),))
    initial = ModuleGraphState(jnp.asarray(0,jnp.int32),{
        "input":channel.initial_value(),"output":ModuleValue((),{"queued":np.asarray(False)})})
    final,trace = CompiledModuleGraph(graph,ticks=3,physics_dt_s=.01).run(initial,{}, {"input":{},"output":{}})
    np.testing.assert_allclose(trace.outputs["input"]["value"],3)
    assert sink.published==3 and sink.dropped==2
    tick,time,values = sink.take()
    assert tick==2 and time==pytest.approx(.02) and values["value"]==3
    assert sink.take() is None


def test_world_boundary_and_native_replay():
    from aerodrome.composition import EntitySpec,WorldSpec,build_world
    from aerodrome.composition.graph_assembly import GraphAssembly
    from aerodrome.runners.world_io import WorldIO
    channel = InputChannel((port(),))
    module = input_module("input",(port(),))
    graph = ModuleGraph((module,),inputs=(port(),),bindings=(InputBinding("value",("input","value")),))
    world = build_world(WorldSpec((EntitySpec("e",GraphAssembly(graph)),),ticks_per_step=2))
    state = world.reset(0,{"e":{"input":ModuleValue((),{"value":jnp.asarray(0.,jnp.float32)})}})
    params = world.parameters({"e":{"input":{}}})
    sink = OutputChannel((port(),))
    runner = WorldIO(world,{"device":channel},{"e":lambda s:{"value":np.where(s["device"]["io_valid"],s["device"]["value"],np.float32(0.))}},
                     outputs=((sink,lambda s,r:{"value":s.entities[0].modules["input"].outputs["value"]}),))
    channel.publish({"value":np.asarray(4.,np.float32)})
    following,record,actual = runner.step(state,params)
    replay,_ = jax.jit(world.step)(state,actual,params)
    np.testing.assert_allclose(following.entities[0].modules["input"].outputs["value"],4.)
    assert int(replay.tick)==2 and sink.take()[2]["value"]==4
    channel.disconnect()
    following,_,_ = runner.step(following,params)
    assert following.entities[0].modules["input"].outputs["value"]==0


def test_camera_fake_capture_color_shape_failure_and_close():
    class Capture:
        closed = False
        good = True
        def isOpened(self): return True
        def read(self): return (True,np.array([[[1,2,3]]],np.uint8)) if self.good else (False,None)
        def release(self): self.closed=True
    capture = Capture()
    cv = SimpleNamespace(VideoCapture=lambda _:capture,COLOR_BGR2RGB=1,cvtColor=lambda a,code:a[:,:,::-1])
    camera = OpenCVCamera(width=1,height=1)
    camera.open(cv2_module=cv)
    assert camera.read()
    np.testing.assert_array_equal(camera.channel.read()["rgb"],[[[3,2,1]]])
    capture.good=False
    assert not camera.read() and not camera.channel.read()["io_valid"]
    camera.close()
    assert capture.closed


def test_pygame_event_adapter_without_device():
    names = ("KEYDOWN","KEYUP","MOUSEBUTTONDOWN","MOUSEBUTTONUP","MOUSEMOTION","MOUSEWHEEL","WINDOWFOCUSLOST","QUIT","WINDOWFOCUSGAINED")
    pg = SimpleNamespace(**{name:i for i,name in enumerate(names)})
    hub = KeyboardMouse(("up",))
    adapter = PygameInput(hub,{42:"up"},pygame_module=pg)
    adapter.handle([SimpleNamespace(type=pg.KEYDOWN,key=42),SimpleNamespace(type=pg.KEYUP,key=42)])
    assert hub.channel.read()["presses"][0]==1


def test_wrong_payload_and_invalid_close():
    channel = InputChannel((port(),))
    with pytest.raises(ValueError):
        channel.publish({"value":np.asarray(1.,np.float64)})
    sink = OutputChannel((port(),))
    sink.close()
    with pytest.raises(RuntimeError):
        sink.publish({"value":np.asarray(1.,np.float32)},tick=0,time_s=0.)


def test_native_fresh_pulse_once_for_held_chunk():
    channel = InputChannel((port(),))
    channel.publish({"value":np.asarray(5.,np.float32)})
    ports = channel.output_ports
    module = input_module("input",ports)
    graph = ModuleGraph((module,),inputs=ports,bindings=tuple(InputBinding(p.name,("input",p.name)) for p in ports))
    values = channel.read()
    state = ModuleGraphState(jnp.asarray(0,jnp.int32),{"input":channel.initial_value()})
    sequences = {k:jnp.stack([v]*3) for k,v in values.items()}
    final,trace = CompiledModuleGraph(graph,ticks=3,physics_dt_s=.01).native(state,sequences,{"input":{}})
    np.testing.assert_array_equal(trace.outputs["input"]["io_fresh"],[True,False,False])
