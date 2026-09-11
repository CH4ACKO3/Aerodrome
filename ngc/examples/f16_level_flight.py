"""Nonlinear F16 longitudinal trim regulation, with explicit simplifying assumptions."""
import json
import sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aerodrome.models.f16_longitudinal import load_tables,Airframe
from aerodrome.pipelines.f16_trim import design
from aerodrome.composition import EntitySpec,WorldSpec,build_world
from aerodrome.composition.f16_longitudinal import F16LongitudinalAssembly,FlightParameters,FlightState
from aerodrome.core.clock import Schedule
from aerodrome.runners.batch import BatchedWorld


def build_experiment():
    tables = load_tables()
    trim,input0,K,info = design(tables)
    schedule = Schedule(physics_dt_s=.01,control_every=2)
    world = build_world(WorldSpec((EntitySpec("f16",F16LongitudinalAssembly()),),schedule,ticks_per_step=2))
    batch = BatchedWorld(world,initial_axes=0,input_axes=None)
    perturbation = jnp.array([5.,jnp.deg2rad(1),jnp.deg2rad(.5),jnp.deg2rad(2),30.])
    initials = jnp.stack((trim,trim+perturbation,trim-perturbation))
    conditions = {"f16":FlightState(initials,jnp.broadcast_to(input0,(3,2)))}
    state = batch.reset(42,[0,1,2],conditions)
    parameters = world.parameters({"f16":FlightParameters(trim,input0,K,Airframe())},resources=tables)
    return batch,state,parameters,trim,input0,K,info


def main():
    jax.config.update("jax_enable_x64",True)
    batch,state,parameters,trim,input0,K,info = build_experiment()
    # Record at the start of each decision interval, including initial perturbations.
    def project(w,trace):
        record = trace.entities[0]
        return {"state":record.aircraft[0],"control":record.applied[0],"saturated":record.saturated[0]}
    run = jax.jit(lambda s,p:batch.rollout_constant(s,(trim,),p,steps=6000,record=project))
    final,records = jax.device_get(run(state,parameters))
    truth = np.asarray(records["state"])
    end = np.asarray(final.world.entities[0].aircraft)
    error = end-np.asarray(trim)
    if not np.all(np.isfinite(truth)):
        raise RuntimeError("trajectory left the data domain or became nonfinite")
    limits = np.array([.1,np.deg2rad(.05),np.deg2rad(.02),np.deg2rad(.05),.2])
    if np.any(np.abs(error)>limits):
        raise RuntimeError(f"120 s recovery failed: {error}")
    time = np.arange(len(truth))*.02
    folder = Path("artifacts/f16_level_flight")
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"trajectory.npz",time_s=time,**records,final_state=end,trim_state=trim,trim_input=input0,K=K,**info)
    settled = []
    for b in range(3):
        outside = np.flatnonzero(np.any(np.abs(truth[:,b]-np.asarray(trim)) > limits,axis=1))
        settled.append(float(time[outside[-1]+1]) if len(outside) and outside[-1]+1<len(time) else (None if len(outside) else 0.))
    fig,axes = plt.subplots(4,2,figsize=(12,12),constrained_layout=True)
    labels = ["Trim", "+ disturbance", "- disturbance"]
    for ax,column,label,scale in zip(axes.flat[:5],range(5),["Airspeed (m/s)","Angle of attack (deg)","Pitch rate (deg/s)","Pitch attitude (deg)","Altitude (m)"],[1,180/np.pi,180/np.pi,180/np.pi,1],strict=True):
        for b in range(3):
            ax.plot(time,truth[:,b,column]*scale,label=labels[b],linewidth=1.3)
        ax.axhline(float(trim[column])*scale,color="black",linestyle="--",linewidth=.7)
        ax.set(xlabel="Time (s)",ylabel=label)
        ax.grid(alpha=.25)
    ax = axes.flat[5]
    for b in range(3):
        ax.plot(time,np.rad2deg(records["control"][:,b,0]),label=labels[b])
    ax.set(xlabel="Time (s)",ylabel="Elevator (deg)")
    ax.grid(alpha=.25)
    ax = axes.flat[6]
    for b in range(3):
        ax.plot(time,records["control"][:,b,1]/1000,label=labels[b])
    ax.set(xlabel="Time (s)",ylabel="Thrust (kN)")
    ax = axes.flat[7]
    for b in range(3):
        ax.plot(time,np.rad2deg(truth[:,b,3]-truth[:,b,1]),label=labels[b])
    ax.set(xlabel="Time (s)",ylabel="Flight-path angle (deg)")
    for ax in axes.flat:
        ax.set_xlim(0,40)
        ax.grid(alpha=.25)
    axes.flat[0].legend()
    fig.suptitle("F-16 longitudinal level-flight regulation | ISRL/NASA1538 tables\n120 s simulation, first 40 s shown | ideal full-state feedback and actuators")
    fig.savefig(folder/"flight.png",dpi=150)
    plt.close(fig)
    summary = dict(model="ISRL/NASA1538 F16 symmetric longitudinal reduction; fixed LEF=0 deg",
                   device=str(jax.devices()[0]),gil_enabled=sys._is_gil_enabled(),duration_s=120.,
                   physics_dt_s=.01,control_dt_s=.02,trim_state=np.asarray(trim).tolist(),
                   trim_state_order=["V_mps","alpha_rad","q_radps","theta_rad","h_m"],
                   trim_elevator_deg=float(jnp.rad2deg(input0[0])),trim_thrust_N=float(input0[1]),
                   max_trim_residual=float(np.max(np.abs(info["trim_residual"]))),
                   closed_loop_pole_radius=float(np.max(np.abs(info["closed_loop_poles"]))),
                   final_errors=error.tolist(),saturated_decision_steps=np.sum(records["saturated"],axis=0).tolist(),
                   settling_time_s=settled,settling_limits=limits.tolist(),
                   max_abs_pitch_rate_deg_s=np.max(np.abs(np.rad2deg(truth[:,:,2])),axis=0).tolist(),
                   max_abs_altitude_error_m=np.max(np.abs(truth[:,:,4]-float(trim[4])),axis=0).tolist(),
                   assumptions=["beta=p=r=roll=0; planar flight","fixed LEF=0; deep-stall correction omitted as upstream",
                                "NASA Glenn troposphere fit; no wind","ideal full-state navigation, no noise",
                                "ideal instantaneous elevator/thrust with magnitude limits; no engine/actuator dynamics",
                                "local trim regulation only; not full-envelope or flight-qualified"])
    (folder/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
