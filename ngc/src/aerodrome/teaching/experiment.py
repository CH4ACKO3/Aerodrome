from pydantic import BaseModel, ConfigDict, Field

MANIFEST = dict(api_version=1, core_version="0.1.0.dev0", curriculum_version="2026.09",
                experiments=["rigid-velocity-v1"])


class VelocityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    initial_velocity_m_s: float = Field(default=0, ge=-20, le=20)
    target_velocity_m_s: float = Field(default=2, ge=-20, le=20)
    gain_per_s: float = Field(default=1, ge=0.05, le=10)
    mass_kg: float = Field(default=1000, ge=1, le=100000)
    max_force_N: float = Field(default=1000, ge=1, le=100000)


def run_velocity(config):
    """250 feedback steps, each containing two World ticks; no gravity or drag."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from aerodrome.configuration import build_experiment
    from aerodrome.models.rigid_body import BodyLoads
    c = VelocityConfig.model_validate(config)
    built = build_experiment({"runtime": {"dtype": "float32"}, "entities": [{
        "id": "body", "model": {"kind": "rigid_body", "version": "1", "parameters": {
            "mass_kg": c.mass_kg, "velocity_body_m_s": [c.initial_velocity_m_s, 0., 0.],
            "omega_body_rad_s": [0., 0., 0.]}}}]}, base_dir=".")
    def step(state, _):
        velocity = state.entities[0].velocity_body_m_s[0]
        force = jnp.clip(c.mass_kg*c.gain_per_s*(c.target_velocity_m_s-velocity),
                         -c.max_force_N, c.max_force_N)
        loads = BodyLoads(jnp.array([force, 0., 0.], dtype=jnp.float32), jnp.zeros(3, dtype=jnp.float32))
        updated, _ = built.world.step(state, (loads,), built.parameters)
        return updated, (updated.entities[0].velocity_body_m_s[0], force)
    _, (velocity, force) = jax.jit(lambda s: jax.lax.scan(step, s, None, length=250))(built.state)
    values = np.asarray(velocity).tolist()
    return dict(experiment="rigid-velocity-v1", manifest=MANIFEST, config=c.model_dump(),
                dt_s=built.world.step_dt_s, device=str(velocity.device),
                time_s=(np.arange(251)*built.world.step_dt_s).tolist(),
                velocity_m_s=[c.initial_velocity_m_s]+values,
                force_N=np.asarray(force).tolist(), final_error_m_s=c.target_velocity_m_s-values[-1])
