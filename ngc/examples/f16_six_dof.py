"""Trimmed six-DoF F-16 and a three-surface pulse through World; no flight controller."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aerodrome.composition import EntitySpec, WorldSpec, build_world
from aerodrome.composition.f16 import F16Assembly
from aerodrome.core.airdata import airdata
from aerodrome.core.clock import Schedule
from aerodrome.core.rotations import quaternion_to_euler321
from aerodrome.models import f16
from aerodrome.pipelines.f16_six_dof_trim import trim


def simulate():
    tables, parameters = f16.load_tables(), f16.default_parameters()
    initial, controls = trim(tables, parameters)
    dt, steps = .01, 400
    world = build_world(WorldSpec((EntitySpec("f16", F16Assembly()),),
                                  Schedule(physics_dt_s=dt), ticks_per_step=1))
    state = world.reset(0, {"f16": initial})
    packed = world.parameters({"f16": parameters}, resources=tables)

    def experiment(amplitude):
        def advance(s, tick):
            pulse = amplitude*((tick >= 50) & (tick < 150))
            applied = controls._replace(elevator_rad=controls.elevator_rad-jnp.deg2rad(.25)*pulse,
                                        aileron_rad=controls.aileron_rad+jnp.deg2rad(2.)*pulse,
                                        rudder_rad=controls.rudder_rad+jnp.deg2rad(.5)*pulse)
            following, _ = world.step(s, (applied,), packed)
            return following, following.entities[0]
        _, trajectory = jax.lax.scan(advance, state, jnp.arange(steps))
        return jax.tree.map(lambda first, rest: jnp.concatenate((first[None], rest)), initial, trajectory)

    trajectories = jax.jit(jax.vmap(experiment))(jnp.array([0., 1.]))
    trajectories = jax.device_get(trajectories)
    if not all(np.isfinite(x).all() for x in trajectories):
        raise RuntimeError("F16 trajectory left its data domain")
    angles = np.asarray(jax.vmap(jax.vmap(quaternion_to_euler321))(trajectories.attitude))
    air = jax.vmap(jax.vmap(airdata))(trajectories.velocity_body_m_s)
    residual = f16.rhs(initial, controls, tables, parameters)
    summary = dict(duration_s=steps*dt, physics_dt_s=dt, cases=["trim", "surface pulse"],
                   pulse_interval_s=[.5, 1.5], pulse_elevator_aileron_rudder_deg=[-.25, 2., .5],
                   trim_controls={name: float(value) for name, value in controls._asdict().items()},
                   max_trim_acceleration_residual=float(max(np.max(np.abs(residual.velocity_body_m_s)),
                                                           np.max(np.abs(residual.omega_body_rad_s)))),
                   max_trim_altitude_drift_m=float(np.max(np.abs(trajectories.position_ned_m[0, :, 2]+3000.))),
                   final_euler_deg=np.rad2deg(angles[:, -1]).tolist(),
                   max_abs_rates_deg_s=np.max(np.abs(np.rad2deg(trajectories.omega_body_rad_s)), axis=1).tolist(),
                   assumptions=["fixed mass, local flat-earth NED, no wind",
                                "actual surface deflections; thrust at COM; no actuator/engine dynamics",
                                "open-loop pulse, no attitude recovery controller"])
    records = dict(time_s=np.arange(steps+1)*dt, **trajectories._asdict(), euler_rad=angles,
                   speed_m_s=np.asarray(air.speed_m_s), alpha_rad=np.asarray(air.alpha_rad),
                   beta_rad=np.asarray(air.beta_rad))
    return records, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/f16_six_dof"))
    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    records, summary = simulate()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/"trajectory.npz", **records)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    for row, label in enumerate(("Roll", "Pitch", "Yaw")):
        for case, name in enumerate(summary["cases"]):
            axes[row, 0].plot(records["time_s"], np.rad2deg(records["euler_rad"][case, :, row]), label=name)
            axes[row, 1].plot(records["time_s"], np.rad2deg(records["omega_body_rad_s"][case, :, row]), label=name)
        axes[row, 0].set_ylabel(f"{label} attitude (deg)")
        axes[row, 1].set_ylabel(f"Body {('p', 'q', 'r')[row]} rate (deg/s)")
    for ax in axes.flat:
        ax.axvspan(.5, 1.5, color="gray", alpha=.12)
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=.25)
    axes[0, 0].legend()
    fig.suptitle("F-16 six-DoF airframe | trim and open-loop surface pulse\n"
                 "ISRL tables, ideal surfaces and external thrust; no recovery controller")
    fig.savefig(args.output/"flight.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
