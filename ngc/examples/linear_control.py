"""Host control design -> compiled LTI World; bounded CPU/device benchmark.

Run in a quiet environment. Comparisons use identical inputs and output samples.
Compilation, transfer and steady-state execution are reported separately.
"""
import argparse
import json
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter
import control as ct
import jax
import jax.numpy as jnp
import numpy as np
from aerodrome.adapters.control import from_control,to_control
from aerodrome.adapters.external import Port
from aerodrome.composition import EntitySpec,WorldSpec,build_world
from aerodrome.composition.graph_assembly import GraphAssembly
from aerodrome.composition.module_graph import ModuleGraph,ModuleValue,InputBinding
from aerodrome.core.clock import Schedule
from aerodrome.models.linear import rollout
from aerodrome.runners.batch import BatchedWorld


def elapsed(function):
    start = perf_counter()
    result = jax.block_until_ready(function())
    return result,perf_counter()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype",choices=("float32","float64"),default="float64")
    parser.add_argument("--batches",nargs="+",type=int,default=[1,64,256])
    parser.add_argument("--steps",type=int,default=512)
    parser.add_argument("--repeats",type=int,default=5)
    args = parser.parse_args()
    if min(*args.batches,args.steps,args.repeats) < 1:
        parser.error("batch sizes, steps and repeats must be positive")
    jax.config.update("jax_enable_x64",True)
    dt = .01
    def prepare():
        model = from_control(ct.tf([3.],[1.,.8,.4],dt=0),sample_time=dt,dtype=args.dtype)
        jax.block_until_ready(model.parameters)
        return model
    model,setup_s = elapsed(prepare)
    reference = to_control(model)
    u,y = Port("u","rad",(1,),"body",dtype=args.dtype),Port("y","rad",(1,),"body",dtype=args.dtype)
    graph = ModuleGraph((model.module("plant",u,y),),inputs=(u,),bindings=(InputBinding("u",("plant","u")),))
    world = build_world(WorldSpec((EntitySpec("plane",GraphAssembly(graph)),),Schedule(physics_dt_s=dt),ticks_per_step=1))
    batch = BatchedWorld(world)
    x0,y0 = model.initialize()
    conditions = {"plane":{"plant":ModuleValue(x0,{"y":y0})}}
    parameters = world.parameters({"plane":{"plant":model.parameters}})
    records = []
    for size in args.batches:
        state = batch.reset(42,np.arange(size),conditions)
        host_inputs = np.sin(np.arange(args.steps)[:,None,None]*.03+np.arange(size)[None,:,None]*.01).astype(args.dtype)
        inputs,transfer_s = elapsed(lambda:jax.device_put(host_inputs))
        run = jax.jit(lambda s,u,p:batch.rollout(s,({"u":u},),p,steps=args.steps,
                      record=lambda w,t:w.entities[0].modules["plant"].outputs["y"]))
        executable,compile_s = elapsed(lambda:run.lower(state,inputs,parameters).compile())
        elapsed(lambda:executable(state,inputs,parameters))
        times = [elapsed(lambda:executable(state,inputs,parameters))[1] for _ in range(args.repeats)]
        result = executable(state,inputs,parameters)
        jax.block_until_ready(result)
        actual,download_s = elapsed(lambda:jax.device_get(result[1]))

        # Minimal numerical kernel reference quantifies adapter/World overhead.
        direct = jax.jit(jax.vmap(rollout,in_axes=(0,1,None),out_axes=(0,1)))
        initial = jnp.zeros((size,x0.size),dtype=args.dtype)
        direct_executable,direct_compile_s = elapsed(lambda:direct.lower(initial,inputs,model.parameters).compile())
        direct_result,_ = elapsed(lambda:direct_executable(initial,inputs,model.parameters))
        direct_times = [elapsed(lambda:direct_executable(initial,inputs,model.parameters))[1] for _ in range(args.repeats)]
        np.testing.assert_allclose(actual,direct_result[1],rtol=5e-5 if args.dtype == "float32" else 1e-11,atol=1e-7 if args.dtype == "float32" else 1e-12)

        def host_run():
            return np.stack([ct.forced_response(reference,T=np.arange(args.steps)*dt,
                            U=host_inputs[:,i,:].T,X0=np.asarray(x0),squeeze=False).outputs.T
                             for i in range(size)],axis=1)
        expected = host_run()  # Warm up before recording the CPU reference.
        # python-control promotes its matrix arithmetic to float64. For float32
        # accumulated roundoff near zero crossings needs a trajectory-scale bound,
        # not a relative error against each nearly-zero sample.
        error_limit = (64*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(expected))))
                       if args.dtype == "float32" else 1e-11)
        assert actual.dtype == np.dtype(args.dtype)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=error_limit)
        host_times = [elapsed(host_run)[1] for _ in range(3)]
        record = dict(batch=size,steps=args.steps,compile_s=compile_s,input_transfer_s=transfer_s,
                      output_transfer_s=download_s,warm_seconds=times,warm_median_s=median(times),
                      direct_compile_s=direct_compile_s,direct_warm_median_s=median(direct_times),
                      control_serial_median_s=median(host_times),
                      world_steps_per_s=size*args.steps/median(times),
                      reference_absolute_error_limit=float(error_limit),
                      max_absolute_error=float(np.max(np.abs(actual-expected))))
        records.append(record)
        print(json.dumps(record),flush=True)
    summary = dict(python=sys.version,control=ct.__version__,jax=jax.__version__,platform=platform.platform(),
                   device=str(jax.devices()[0]),dtype=args.dtype,gil_enabled=sys._is_gil_enabled(),
                   host_model_setup_s=setup_s,records=records,
                   caveat="Warm synchronized timings; serial python-control is a functional reference, not an optimized batch baseline. No speed guarantee.")
    folder = Path("artifacts/linear_control")
    folder.mkdir(parents=True,exist_ok=True)
    (folder/f"benchmark-{args.dtype}.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if sys._is_gil_enabled():
        raise RuntimeError("a dependency enabled the GIL")


if __name__ == "__main__":
    main()
