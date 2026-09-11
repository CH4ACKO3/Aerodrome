"""CPU runnable architecture example. No performance or aircraft fidelity claim."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from aerodrome.core.clock import Schedule
from aerodrome.core.signals import PitchState, GaussianState, PitchGoal
from aerodrome.models.pitch import PitchParameters
from aerodrome.navigation.kalman import discretize
from aerodrome.control.pid import PIDParameters
from aerodrome.systems.pitch_loop import Parameters, initialize, make_step
from aerodrome.runners.native import run_scan


def build_case(seed=42):
    schedule = Schedule()
    nominal = PitchParameters(*map(jnp.asarray, (0.4, 0.8, 3.0)))
    params = Parameters(
        true_model=nominal,
        navigation=discretize(nominal, schedule.physics_dt_s, 1e-5, 0.005),
        control=PIDParameters(*map(jnp.asarray, (2.0, 0.8, 0.8, 0.35))),
        sensor_std_rad=jnp.asarray(0.005), reference_rate_rad_s=jnp.asarray(0.1),
    )
    truth = PitchState(jnp.asarray(0.05), jnp.asarray(0.0))
    # Deliberately distinct initial belief, not copied from the true state.
    prior = GaussianState(PitchState(jnp.asarray(0.), jnp.asarray(0.)), jnp.diag(jnp.array([0.01, 0.04])))
    return schedule, params, initialize(truth, prior, jax.random.key(seed)), PitchGoal(jnp.asarray(0.1))


def main():
    # Application chooses precision. Importing the package does not change it.
    jax.config.update("jax_enable_x64", True)
    schedule, params, initial, goal = build_case()
    step = make_step(schedule)
    run = jax.jit(lambda state, p: run_scan(step, state, goal, p, steps=1000))
    final, trace = jax.device_get(run(initial, params))
    result = {
        "model": "illustrative two-state pitch model; not calibrated aircraft",
        "jax": jax.__version__, "device": str(jax.devices()[0]),
        "precision": str(trace.truth.pitch_rad.dtype), "seed": 42,
        "steps": int(final.tick), "physics_dt_s": schedule.physics_dt_s,
        "tracking_rmse_rad": float(np.sqrt(np.mean((trace.truth.pitch_rad - trace.reference.pitch_rad)**2))),
        "estimation_rmse_rad": float(np.sqrt(np.mean((trace.truth.pitch_rad - trace.navigation.estimate.pitch_rad)**2))),
        "final_pitch_rad": float(final.truth.pitch_rad),
    }
    folder = Path("artifacts")
    folder.mkdir(exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(folder / "trajectory.npz", time_s=trace.time_s,
             truth_pitch_rad=trace.truth.pitch_rad, measurement_pitch_rad=trace.measurement.pitch_rad,
             measurement_valid=trace.measurement.valid, estimate_pitch_rad=trace.navigation.estimate.pitch_rad,
             reference_pitch_rad=trace.reference.pitch_rad, elevator_rad=trace.actuator.elevator_rad)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
