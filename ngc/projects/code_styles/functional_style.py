"""Function style: explicit inputs and a result, with I/O at the entry point."""
import argparse
import json
from pathlib import Path

import numpy as np
from aerodrome.models.point_mass import PointMassState, step


def simulate(*, initial_velocity_m_s=0., target_m_s=2., gain_per_s=1.,
             max_acceleration=1., dt_s=.02, steps=250):
    """Return fresh arrays; no hidden state, files, printing or global changes.

    This is functional organization, not a claim that this Python loop is a
    compiled JAX scan. The shared physical step can be used with either array
    backend. Explicit arguments make parameter sweeps easy to write.
    """
    state = PointMassState(0., initial_velocity_m_s)
    position, velocity, acceleration = [state.position_m], [state.velocity_m_s], []
    for _ in range(steps):
        command = np.clip(gain_per_s*(target_m_s-state.velocity_m_s),
                          -max_acceleration, max_acceleration)
        state = step(state, command, dt_s)
        position.append(state.position_m)
        velocity.append(state.velocity_m_s)
        acceleration.append(command)
    return dict(time_s=np.arange(steps+1)*dt_s, position_m=np.asarray(position),
                velocity_m_s=np.asarray(velocity), acceleration_m_s2=np.asarray(acceleration))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/styles/functional"))
    args = parser.parse_args()
    records = simulate()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/"trajectory.npz", **records)
    summary = dict(style="functional", target_m_s=2.,
                   final_velocity_m_s=float(records["velocity_m_s"][-1]), steps=250, dt_s=.02)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
