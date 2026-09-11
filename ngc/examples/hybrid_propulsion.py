"""Swap a JAX engine partition for a protocol fixture; no MATLAB installation used.

    Engine: dT/dt=(throttle*Tmax-T)/tau.
    Body: dx/dt=v; dv/dt=T/m (one-dimensional teaching example, not an aircraft).
    Both partitions use exact interval solutions with held boundary inputs.
"""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from aerodrome.adapters.external import Port, ExternalSample, Capabilities
from aerodrome.adapters.functional import FunctionalComponent
from aerodrome.runners.cosimulation import CoSimulationRunner, Connection


THROTTLE = Port("throttle", "1", (), "scalar", quantity="throttle_fraction")
THRUST = Port("thrust", "N", (), "body", quantity="axial_force", reference_point="engine_mount")
POSITION = Port("position", "m", (), "inertial", quantity="position")
SPEED = Port("speed", "m/s", (), "inertial", quantity="velocity")


def jax_engine():
    def initialize(p):
        return jnp.asarray(0.), {"thrust": jnp.asarray(0.)}
    @jax.jit
    def transition(thrust, inputs, p, dt):
        target = inputs["throttle"] * p["max_thrust"]
        thrust = target + (thrust - target) * jnp.exp(-dt / p["tau"])
        return thrust, {"thrust": thrust}
    return FunctionalComponent((THROTTLE,), (THRUST,), initialize, transition)


class ExternalEngineFixture:
    """In-process NumPy stand-in for an external solver, NOT a MATLAB adapter."""
    inputs, outputs = (THROTTLE,), (THRUST,)
    capabilities = Capabilities(True, False, False, False)

    def initialize(self, start_time_s, parameters):
        self.start = self.time = start_time_s
        self.params = parameters
        self.thrust, self.closed = 0., False
        return ExternalSample(self.time, {"thrust": np.asarray(self.thrust)})

    def advance_to(self, target_time_s, held_inputs):
        target = float(held_inputs["throttle"]) * self.params["max_thrust"]
        self.thrust = target + (self.thrust - target) * np.exp(-(target_time_s-self.time)/self.params["tau"])
        self.time = target_time_s
        return ExternalSample(self.time, {"thrust": np.asarray(self.thrust)})

    def reset(self):
        return self.initialize(self.start, self.params)

    def close(self):
        self.closed = True


def jax_body():
    def initialize(p):
        state = (jnp.asarray(0.), jnp.asarray(0.))
        return state, {"position": state[0], "speed": state[1]}
    @jax.jit
    def transition(state, inputs, p, dt):
        position, speed = state
        acceleration = inputs["thrust"] / p["mass"]
        state = (position + speed*dt + acceleration*dt*dt/2, speed + acceleration*dt)
        return state, {"position": state[0], "speed": state[1]}
    return FunctionalComponent((THRUST,), (POSITION, SPEED), initialize, transition)


def run(engine_factory, *, dt=0.02, steps=100):
    with CoSimulationRunner(
        {"engine": engine_factory(), "body": jax_body()},
        (Connection(("engine", "thrust"), ("body", "thrust")),),
        communication_dt_s=dt, external_inputs=(("engine", "throttle"),),
    ) as runner:
        runner.initialize({"engine": {"max_thrust": 1000., "tau": 0.5}, "body": {"mass": 100.}})
        trajectory = []
        for _ in range(steps):
            sample = runner.step({("engine", "throttle"): np.asarray(0.7)})
            trajectory.append([sample["body"].time_s,
                               float(sample["body"].values["position"]),
                               float(sample["body"].values["speed"]),
                               float(sample["engine"].values["thrust"])])
        return np.asarray(trajectory)


def main():
    jax.config.update("jax_enable_x64", True)
    native, external = run(jax_engine), run(ExternalEngineFixture)
    np.testing.assert_allclose(native, external, rtol=1e-12, atol=1e-12)
    report = {"example": "1D propulsion; protocol fixture, no MATLAB",
              "communication_dt_s": 0.02, "steps": 100,
              "columns": ["time_s", "position_m", "speed_m_s", "thrust_N"],
              "max_abs_difference_by_column": np.max(np.abs(native-external), axis=0).tolist()}
    folder = Path("artifacts/hybrid_propulsion")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(folder / "trajectory.npz", jax_engine=native, external_fixture=external)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
