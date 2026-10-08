import jax
import jax.numpy as jnp
import numpy as np
import pytest
from aerodrome.analysis import python_control,matlab_compat,backend_status
from aerodrome.adapters.control import from_control, from_descriptor, from_matrices
from aerodrome.adapters.linear_delay import with_input_delay
from aerodrome.models.linear import rollout
from aerodrome.adapters.external import Port
from aerodrome.composition.module_graph import ModuleGraph,ModuleGraphState,ModuleValue,InputBinding
from aerodrome.runners.module_compiler import CompiledModuleGraph

ct = pytest.importorskip("control")


def test_analysis_connections_frequency_and_design():
    api = python_control()
    G = api.zpk([],[-1.],2.)
    closed = api.feedback(G,1)
    np.testing.assert_allclose(api.dcgain(closed),2/3)
    np.testing.assert_allclose(api.evalfr(closed,1j),2/(3+1j))
    K,P,poles = api.lqr(np.array([[-1.]]),np.ones((1,1)),np.ones((1,1)),np.ones((1,1)),method="scipy")
    np.testing.assert_allclose(-2*P-P@P+1,0,atol=1e-12)
    assert np.all(np.real(poles)<0)
    values,times,states = matlab_compat().lsim(api.ss(closed),U=np.ones(20),T=np.arange(20)*.1)
    assert values.shape == (20,) and times.shape == (20,)
    assert backend_status()["packages"]["control"] == ct.__version__


@pytest.mark.parametrize("dt",[0,.1])
def test_coupled_mimo_tf_matches_individual_channels(dt):
    G = ct.tf([[[1.,2.],[3.]],[[2.],[0.]]],
              [[[1.,3.,2.],[1.,4.]],[[1.],[1.]]],dt=dt)
    model = from_control(G,sample_time=.1)
    u = np.stack((np.sin(np.arange(30)*.2),np.cos(np.arange(30)*.1)),axis=1)
    _,y = jax.jit(rollout)(jnp.zeros(model.parameters.A.shape[0]),jnp.asarray(u),model.parameters)
    expected = np.zeros((30,2))
    for row in range(2):
        for col in range(2):
            channel = ct.tf2ss(G[row,col],method="scipy")
            if dt == 0:
                channel = ct.sample_system(channel,.1)
            expected[:,row] += ct.forced_response(channel,T=np.arange(30)*.1,U=u[:,col]).outputs
    np.testing.assert_allclose(y,expected,rtol=1e-10,atol=1e-12)


@pytest.mark.parametrize("dt",[None,.1])
def test_invertible_descriptor_preserves_dynamics(dt):
    A,B,C,D,E = np.diag([-2.,-3.]),np.ones((2,1)),np.ones((1,2)),np.zeros((1,1)),np.array([[2.,1.],[0.,3.]])
    model = from_descriptor(A,B,C,D,E,dt=dt,sample_time=.1)
    reference = from_matrices(np.linalg.solve(E,A),np.linalg.solve(E,B),C,D,dt=dt,sample_time=.1)
    for a,b in zip(model.parameters,reference.parameters,strict=True):
        np.testing.assert_allclose(a,b)


def delay_case(steps):
    model = from_matrices([[.5]],[[1.,2.]],[[1.]],[[.3,.4]],dt=.1)
    delayed = with_input_delay(model,steps)
    u,y = Port("u","1",(2,),"test"),Port("y","1",(1,),"test")
    graph = ModuleGraph((delayed.module("plant",u,y),),inputs=(u,),bindings=(InputBinding("u",("plant","u")),))
    x,out = delayed.initialize()
    return delayed,graph,ModuleGraphState(jnp.int32(0),{"plant":ModuleValue(x,{"y":out})})


@pytest.mark.parametrize("steps",[(0,0),(0,2),(1,3)])
def test_ring_delay_matches_shifted_inputs_batch_and_chunking(steps):
    model,graph,state = delay_case(steps)
    u = jnp.arange(24.,dtype=float).reshape(12,2)
    p = {"plant":model.parameters}
    run = CompiledModuleGraph(graph,ticks=12,physics_dt_s=.1).native
    final,trace = run(state,{"u":u},p)
    shifted = np.zeros_like(u)
    for i,d in enumerate(steps):
        shifted[d:,i] = np.asarray(u)[:12-d,i]
    expected,outputs = rollout(jnp.zeros(1),jnp.asarray(shifted),model.parameters)
    np.testing.assert_allclose(trace.outputs["plant"]["y"],outputs)
    np.testing.assert_allclose(final.modules["plant"].state.plant,expected)
    middle,a = CompiledModuleGraph(graph,ticks=5,physics_dt_s=.1).native(state,{"u":u[:5]},p)
    end,b = CompiledModuleGraph(graph,ticks=7,start_tick=5,physics_dt_s=.1).native(middle,{"u":u[5:]},p)
    for x,y in zip(jax.tree.leaves((final,trace)),jax.tree.leaves((end,jax.tree.map(lambda x,y:jnp.concatenate((x,y)),a,b))),strict=True):
        np.testing.assert_allclose(x,y)
    batched = jax.jit(jax.vmap(run,in_axes=(0,None,None)))(jax.tree.map(lambda x:jnp.stack((x,x)),state),{"u":u},p)
    np.testing.assert_allclose(batched[1].outputs["plant"]["y"][1],outputs)


def test_delay_uses_initial_input_history():
    model,_,_ = delay_case((1,2))
    history = jnp.array([[1.,2.],[3.,4.]])
    _,y = model.initialize(history=history)
    np.testing.assert_allclose(y,[.3*3+.4*2])
