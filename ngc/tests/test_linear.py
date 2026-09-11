import sys
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import cont2discrete
from aerodrome.models.linear import LinearParameters, step, rollout
from aerodrome.adapters.control import from_control, from_matrices, to_control
from aerodrome.adapters.external import Port
from aerodrome.composition.module_graph import ModuleGraph, ModuleGraphState, ModuleValue, InputBinding, Wire
from aerodrome.runners.module_compiler import CompiledModuleGraph

ct = pytest.importorskip("control")


def test_transfer_function_import_matches_forced_response_and_labels():
    system = ct.tf([3.],[1.,.8,.4],inputs=["elevator"],outputs=["pitch"],dt=0)
    model = from_control(system,sample_time=.01)
    inputs = np.sin(np.arange(100)*.03)[:,None]
    x0 = jnp.array([.2,-.1])  # Canonical realization coordinates, not physical pitch/q.
    _, values = jax.jit(rollout)(x0,jnp.asarray(inputs),model.parameters)
    discrete = ct.sample_system(ct.tf2ss(system,method="scipy"),.01,method="zoh")
    expected = ct.forced_response(discrete,T=np.arange(100)*.01,U=inputs.T,X0=np.asarray(x0),squeeze=False)
    np.testing.assert_allclose(values,np.asarray(expected.outputs).T,rtol=1e-11,atol=1e-12)
    assert model.input_labels == ("elevator",) and model.output_labels == ("pitch",)
    exported = to_control(model)
    assert exported.input_labels == ["elevator"] and exported.dt == .01
    np.testing.assert_array_equal(exported.A,model.parameters.A)


@pytest.mark.parametrize("method",["zoh","tustin"])
def test_mimo_discretization_and_direct_feedthrough(method):
    A = np.array([[-1.,.2],[0.,-2.]])
    B,C,D = np.eye(2),np.array([[1.,2.],[3.,4.]]),np.array([[.1,.2],[.3,.4]])
    model = from_control(ct.ss(A,B,C,D,dt=0),sample_time=.02,method=method)
    expected = cont2discrete((A,B,C,D),.02,method="bilinear" if method == "tustin" else method)
    for actual,reference in zip(model.parameters,expected[:4],strict=True):
        np.testing.assert_allclose(actual,reference,rtol=1e-12,atol=1e-12)
    x,u = jnp.array([1.,2.]),jnp.array([3.,4.])
    following,y = jax.jit(step)(x,u,model.parameters)
    np.testing.assert_allclose(y,expected[2]@x+expected[3]@u)
    np.testing.assert_allclose(following,expected[0]@x+expected[1]@u)


def test_discrete_import_does_not_discretize_twice(monkeypatch):
    import aerodrome.adapters.control as bridge
    def forbidden(*args,**kwargs):
        raise AssertionError("host conversion called")
    monkeypatch.setattr(bridge,"cont2discrete",forbidden)
    model = from_control(ct.ss([[.9]],[[.2]],[[1.]],[[.3]],dt=.1))
    np.testing.assert_allclose(model.parameters.A,[[.9]])
    with pytest.raises(ValueError,match="resampled"):
        from_control(to_control(model),sample_time=.2)


def test_compiled_execution_has_dynamic_parameters_and_no_host_conversion(monkeypatch):
    import aerodrome.adapters.control as bridge
    model = from_control(ct.tf([1.],[1.,1.]),sample_time=.1)
    def forbidden(*args,**kwargs):
        raise AssertionError("host conversion entered execution")
    monkeypatch.setattr(bridge,"cont2discrete",forbidden)
    monkeypatch.setattr(ct,"tf2ss",forbidden)
    inputs = jnp.ones((10,1))
    executable = jax.jit(rollout).lower(jnp.zeros(1),inputs,model.parameters).compile()
    original = executable(jnp.zeros(1),inputs,model.parameters)
    changed = executable(jnp.zeros(1),inputs,model.parameters._replace(B=2*model.parameters.B))
    np.testing.assert_allclose(changed[0],2*original[0])
    np.testing.assert_allclose(changed[1],2*original[1])
    assert not sys._is_gil_enabled()


def test_batched_parameter_gradients_are_independent():
    model = from_matrices([[.5]],[[1.]],[[1.]],[[0.]],dt=.1)
    actions = jnp.ones((4,1))
    def one(gain):
        p = model.parameters._replace(B=gain.reshape(1,1))
        return rollout(jnp.zeros(1),actions,p)[0][0]
    jac = jax.jit(jax.jacrev(jax.vmap(one)))(jnp.array([1.,2.,3.]))
    np.testing.assert_allclose(jac,np.eye(3)*1.875)


def test_module_batch_multirate_matches_direct_scan():
    from aerodrome.composition import WorldSpec, EntitySpec, WorldParameters, build_world
    from aerodrome.composition.graph_assembly import GraphAssembly
    from aerodrome.runners.batch import BatchedWorld
    model = from_control(ct.tf([3.],[1.,.8,.4]),sample_time=.02)
    port_in,port_out = Port("u","rad",(1,),"body"),Port("y","rad",(1,),"body")
    graph = ModuleGraph((model.module("plant",port_in,port_out,every=2),),inputs=(port_in,),bindings=(InputBinding("u",("plant","u")),))
    world = build_world(WorldSpec((EntitySpec("aircraft",GraphAssembly(graph)),),ticks_per_step=1))
    axes = WorldParameters(({"plant":LinearParameters(None,0,None,None)},),None)
    batch = BatchedWorld(world,parameter_axes=axes)
    x,y = model.initialize()
    initial = {"aircraft":{"plant":ModuleValue(x,{"y":y})}}
    state = batch.reset(42,[10,20,30],initial)
    gains = jnp.array([1.,2.,3.])
    p = model.parameters._replace(B=gains[:,None,None]*model.parameters.B)
    params = world.parameters({"aircraft":{"plant":p}})
    inputs = jnp.arange(36.,dtype=float).reshape(12,3,1)*.01
    final,trace = jax.jit(lambda s,u,p:batch.rollout(s,({"u":u},),p,steps=12,
        record=lambda w,t:w.entities[0].modules["plant"].outputs["y"]))(state,inputs,params)
    for i in range(3):
        expected,outputs = rollout(x,inputs[::2,i],model.parameters._replace(B=p.B[i]))
        np.testing.assert_allclose(trace[:,i],np.repeat(outputs,2,axis=0),atol=1e-12)
        np.testing.assert_allclose(final.world.entities[0].modules["plant"].state[i],expected,atol=1e-12)


def test_module_period_mismatch_and_algebraic_cycle_rejected():
    model = from_matrices([[.9]],[[1.]],[[1.]],[[.2]],dt=.02)
    u,y = Port("u","1",(1,),"test"),Port("y","1",(1,),"test")
    module = model.module("plant",u,y)
    with pytest.raises(ValueError,match="cycle"):
        ModuleGraph((module,),(Wire(("plant","y"),("plant","u")),))
    graph = ModuleGraph((module,),inputs=(u,),bindings=(InputBinding("u",("plant","u")),))
    x,out = model.initialize()
    initial = ModuleGraphState(jnp.int32(0),{"plant":ModuleValue(x,{"y":out})})
    with pytest.raises(ValueError,match="sample period"):
        CompiledModuleGraph(graph,ticks=2,physics_dt_s=.01).native(initial,{"u":jnp.zeros((2,1))},{"plant":model.parameters})


def test_static_gain_and_float32():
    model = from_matrices(np.zeros((0,0)),np.zeros((0,2)),np.zeros((1,0)),[[2.,3.]],dt=.1,dtype="float32")
    x,_ = model.initialize()
    final,y = jax.jit(rollout)(x,jnp.ones((3,2),jnp.float32),model.parameters)
    assert final.shape == (0,) and y.dtype == jnp.float32
    np.testing.assert_allclose(y,5.)


@pytest.mark.parametrize("dt",[None,True])
def test_unspecified_control_timebase_rejected(dt):
    with pytest.raises(ValueError,match="timebase"):
        from_control(ct.ss([[-1.]],[[1.]],[[1.]],[[0.]],dt=dt),sample_time=.1)


def test_invalid_matrices_precision_and_nonproper_tf_rejected():
    with pytest.raises(ValueError,match="dimensions"):
        from_matrices([[1.]],[[1.,2.]],[[1.]],[[0.]],dt=.1)
    with pytest.raises(ValueError,match="finite"):
        from_matrices([[np.nan]],[[1.]],[[1.]],[[0.]],dt=.1)
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64",False)
    try:
        with pytest.raises(ValueError,match="x64"):
            from_matrices([[1.]],[[1.]],[[1.]],[[0.]],dt=.1)
    finally:
        jax.config.update("jax_enable_x64",previous)
    with pytest.raises(ValueError):
        from_control(ct.tf([1.,1.],[1.]),sample_time=.1)


def test_numerical_kernel_does_not_load_control_or_matplotlib():
    import subprocess
    subprocess.run([sys.executable,"-c","import sys; import aerodrome.models.linear; "
                    "assert 'control' not in sys.modules; assert 'matplotlib' not in sys.modules"],check=True)


def test_mimo_tf_realization():
    tf = ct.tf([[[1.],[0.]],[[0.],[1.]]],[[[1.,1.],[1.]],[[1.],[1.,1.]]],dt=0)
    model = from_control(tf,sample_time=.1)
    assert model.parameters.A.shape == (2,2)
    assert model.conversion_notes
    np.testing.assert_allclose(model.parameters.A,np.eye(2)*np.exp(-.1))
