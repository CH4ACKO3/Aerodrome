"""F-16 attitude recovery with simple PD feedback and fixed-trim comparison."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aerodrome.composition import EntitySpec, WorldSpec, build_world
from aerodrome.composition.f16 import F16Assembly
from aerodrome.control.f16_attitude import Gains, command
from aerodrome.core.clock import Schedule
from aerodrome.core.rotations import euler321_to_quaternion, quaternion_to_euler321
from aerodrome.models import f16
from aerodrome.pipelines.f16_six_dof_trim import trim


def simulate():
    tables, parameters = f16.load_tables(), f16.default_parameters()
    reference, trim_controls = trim(tables, parameters)
    ref_angles = f16.BODY.euler_angles(reference)
    schedule = Schedule(physics_dt_s=.01, control_every=2)
    world = build_world(WorldSpec((EntitySpec("f16", F16Assembly()),), schedule, ticks_per_step=2))
    packed = world.parameters({"f16": parameters}, resources=tables)
    steps = 1000  # 20 s, controller 50 Hz, physics 100 Hz.

    def experiment(sign, feedback):
        initial = reference._replace(
            attitude=euler321_to_quaternion(ref_angles+jnp.deg2rad(sign*jnp.array([5., 2., 0.]))),
            omega_body_rad_s=jnp.deg2rad(sign*jnp.array([1., .5, 1.])))
        state = world.reset(0, {"f16": initial})

        def advance(s, _):
            controls = jax.lax.cond(feedback,
                lambda: command(s.entities[0], reference, trim_controls, parameters), lambda: trim_controls)
            following, _ = world.step(s, (controls,), packed)
            return following, (following.entities[0], jnp.array(controls))

        _, (trajectory, controls) = jax.lax.scan(advance, state, None, length=steps)
        trajectory = jax.tree.map(lambda first, rest: jnp.concatenate((first[None], rest)), initial, trajectory)
        return trajectory, controls

    trajectory, controls = jax.jit(jax.vmap(experiment))(
        jnp.array([1., -1., 1.]), jnp.array([True, True, False]))
    trajectory, controls = jax.device_get((trajectory, controls))
    if not all(np.all(np.isfinite(x)) for x in (*trajectory, controls)):
        raise RuntimeError("attitude experiment left the aerodynamic data domain")
    angles = np.asarray(jax.vmap(jax.vmap(quaternion_to_euler321))(trajectory.attitude))
    errors = angles - np.asarray(ref_angles)
    speeds = np.linalg.norm(trajectory.velocity_body_m_s, axis=-1)  # no wind in this example
    time = np.arange(steps+1)*schedule.control_dt_s
    # A joint attitude/rate/speed band must remain satisfied through the end.
    inside = ((np.max(np.abs(np.rad2deg(errors[:, :, :2])), axis=-1) < .1)
              & (np.max(np.abs(np.rad2deg(trajectory.omega_body_rad_s)), axis=-1) < .1)
              & (np.abs(speeds-150.) < .1))
    settled = []
    for band in inside:
        outside = np.flatnonzero(~band)
        settled.append(0. if not len(outside) else
                       float(time[outside[-1]+1]) if outside[-1] < steps else None)
    summary = dict(cases=["PD + disturbance", "PD - disturbance", "Fixed trim + disturbance"],
                   duration_s=20., physics_dt_s=schedule.physics_dt_s, control_dt_s=schedule.control_dt_s,
                   initial_roll_pitch_yaw_offset_deg=[5., 2., 0.], initial_rates_deg_s=[1., .5, 1.],
                   gains=Gains()._asdict(), trim_controls={k: float(v) for k, v in trim_controls._asdict().items()},
                   final_roll_pitch_error_deg=np.rad2deg(errors[:, -1, :2]).tolist(),
                   final_rates_deg_s=np.rad2deg(trajectory.omega_body_rad_s[:, -1]).tolist(),
                   final_speed_error_m_s=(speeds[:, -1]-150.).tolist(),
                   final_height_change_m=(-trajectory.position_ned_m[:, -1, 2]-3000.).tolist(),
                   settling_time_s=settled, settling_band="roll/pitch <0.1 deg; p/q/r <0.1 deg/s; speed <0.1 m/s",
                   assumptions=["local gains at 150 m/s and 3000 m; no wind",
                                "ideal state feedback, direct surfaces and external thrust",
                                "no altitude or heading hold; no actuator or engine dynamics"])
    records = dict(time_s=time, control_time_s=time[:-1], **trajectory._asdict(),
                   euler_error_rad=errors, speed_m_s=speeds, controls=controls)
    return records, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/f16_attitude_hold"))
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
    series = [np.rad2deg(records["euler_error_rad"][:, :, 0]),
              np.rad2deg(records["euler_error_rad"][:, :, 1]),
              np.rad2deg(records["omega_body_rad_s"][:, :, 2]), records["speed_m_s"]-150.,
              np.rad2deg(records["controls"][:, :, 1]), np.rad2deg(records["controls"][:, :, 0])]
    labels = ["Roll error (deg)", "Pitch error (deg)", "Yaw rate r (deg/s)",
              "Airspeed error (m/s)", "Aileron (deg)", "Elevator (deg)"]
    for i, (ax, values, label) in enumerate(zip(axes.flat, series, labels, strict=True)):
        times = records["time_s"] if i < 4 else records["control_time_s"]
        for case, name in enumerate(summary["cases"]):
            ax.plot(times, values[case], label=name, linestyle="--" if case == 2 else "-")
        ax.set(xlabel="Time (s)", ylabel=label)
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("F-16 attitude PD + yaw damper + airspeed P\n"
                 "Same positive disturbance with/without feedback; negative disturbance also shown")
    fig.savefig(args.output/"control.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
