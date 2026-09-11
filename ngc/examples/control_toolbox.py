"""Expanded model coverage: MIMO TF, descriptor E, and exact sampled delays."""
import json
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from aerodrome.analysis import python_control,matlab_compat,backend_status
from aerodrome.adapters.control import from_control,from_descriptor,to_control
from aerodrome.adapters.linear_delay import with_input_delay
from aerodrome.adapters.external import Port
from aerodrome.composition.module_graph import ModuleGraph,ModuleGraphState,ModuleValue,InputBinding
from aerodrome.runners.module_compiler import CompiledModuleGraph
from aerodrome.telemetry import PerformanceProbe


def main():
    jax.config.update("jax_enable_x64",True)
    ct = python_control()
    G = ct.tf([[[1.],[2.]],[[0.],[1.]]],[[[1.,1.],[1.,2.]],[[1.],[1.,3.]]],dt=0)
    model = from_control(G,sample_time=.02)
    delayed = with_input_delay(model,(0,3))
    u,y = Port("u","1",(2,),"normalized"),Port("y","1",(2,),"normalized")
    graph = ModuleGraph((delayed.module("plant",u,y),),inputs=(u,),bindings=(InputBinding("u",("plant","u")),))
    state,initial_y = delayed.initialize()
    initial = ModuleGraphState(jnp.int32(0),{"plant":ModuleValue(state,{"y":initial_y})})
    program = CompiledModuleGraph(graph,ticks=64,physics_dt_s=.02)
    inputs,params = {"u":jnp.ones((64,2))},{"plant":model.parameters}
    probe = PerformanceProbe()
    executable = probe.call("rollout",lambda:program.native.lower(initial,inputs,params).compile(),phase="compile")
    probe.call("rollout",executable,initial,inputs,params,phase="warmup")
    final,trace = probe.call("rollout",executable,initial,inputs,params,phase="warm")
    history = final.modules["plant"].state.history
    descriptor = from_descriptor([[-2.]],[[1.]],[[1.]],[[0.]],[[2.]],sample_time=.02)
    np.testing.assert_allclose(descriptor.parameters.A,[[np.exp(-.02)]])
    K,_,poles = ct.lqr([[-1.]],[[1.]],[[1.]],[[1.]],method="scipy")
    response,time,state_trace = matlab_compat().lsim(to_control(descriptor),U=np.ones(20),T=np.arange(20)*.02)
    folder = Path("artifacts/control_toolbox")
    folder.mkdir(parents=True,exist_ok=True)
    probe.write(folder/"performance.json")
    summary = dict(backend=backend_status(),device=str(jax.devices()[0]),gil_enabled=sys._is_gil_enabled(),
                   mimo_states=int(model.parameters.A.shape[0]),conversion_notes=model.conversion_notes,
                   delay_steps=delayed.steps,delay_history_shape=list(history.shape),
                   output_shape=list(trace.outputs["plant"]["y"].shape),lqr_gain=np.asarray(K).tolist(),
                   closed_loop_poles=np.asarray(poles).real.tolist(),matlab_style_response_shape=list(response.shape))
    (folder/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
