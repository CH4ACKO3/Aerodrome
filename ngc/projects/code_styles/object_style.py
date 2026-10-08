"""Object style: one experiment owns its configuration and evolving state."""
import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from aerodrome.models.point_mass import PointMassState, step


@dataclass
class VelocityExperiment:
    """A small stateful object; no inheritance tree or controller factory.

    step() advances THIS instance. reset() makes another episode explicit.
    The numerical physics still lives in the shared pure point-mass function.
    """
    target_m_s: float = 2.
    gain_per_s: float = 1.
    max_acceleration: float = 1.
    dt_s: float = .02
    state: PointMassState = field(default_factory=lambda: PointMassState(0., 0.), init=False)

    def reset(self, initial_velocity_m_s=0.):
        self.state = PointMassState(0., initial_velocity_m_s)

    def step(self):
        acceleration = np.clip(self.gain_per_s*(self.target_m_s-self.state.velocity_m_s),
                               -self.max_acceleration, self.max_acceleration)
        self.state = step(self.state, acceleration, self.dt_s)
        return acceleration

    def run(self, steps=250):
        """Continue from the current state; recorded time starts at this call."""
        position, velocity, acceleration = [self.state.position_m], [self.state.velocity_m_s], []
        for _ in range(steps):
            acceleration.append(self.step())
            position.append(self.state.position_m)
            velocity.append(self.state.velocity_m_s)
        return dict(time_s=np.arange(steps+1)*self.dt_s, position_m=np.asarray(position),
                    velocity_m_s=np.asarray(velocity), acceleration_m_s2=np.asarray(acceleration))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/styles/object"))
    args = parser.parse_args()
    experiment = VelocityExperiment()
    experiment.reset()
    records = experiment.run()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output/"trajectory.npz", **records)
    summary = dict(style="object", target_m_s=experiment.target_m_s,
                   final_velocity_m_s=float(experiment.state.velocity_m_s), steps=250, dt_s=experiment.dt_s)
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
