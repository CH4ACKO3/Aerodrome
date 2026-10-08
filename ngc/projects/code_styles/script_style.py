"""Script style: execute the experiment from top to bottom.

Run from ngc: python projects/code_styles/script_style.py --output artifacts/styles/script
This file intentionally has no reusable experiment function or custom class.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from aerodrome.models.point_mass import PointMassState, step

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("artifacts/styles/script"))
args = parser.parse_args()

# All three styles use exactly these conditions. The input is acceleration,
# not force; mass has already been divided out in this one-dimensional example.
dt_s, steps = .02, 250
target_m_s, gain_per_s, max_acceleration = 2., 1., 1.
state = PointMassState(0., 0.)
position, velocity, acceleration = [state.position_m], [state.velocity_m_s], []

for _ in range(steps):
    # P feedback becomes a held input for this complete integration interval.
    command = np.clip(gain_per_s*(target_m_s-state.velocity_m_s),
                      -max_acceleration, max_acceleration)
    state = step(state, command, dt_s)
    position.append(state.position_m)
    velocity.append(state.velocity_m_s)
    acceleration.append(command)

# States include the initial and final instant; commands label interval starts.
args.output.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output/"trajectory.npz", time_s=np.arange(steps+1)*dt_s,
                    position_m=position, velocity_m_s=velocity,
                    acceleration_m_s2=acceleration)
summary = dict(style="script", target_m_s=target_m_s,
               final_velocity_m_s=float(state.velocity_m_s), steps=steps, dt_s=dt_s)
(args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary))
