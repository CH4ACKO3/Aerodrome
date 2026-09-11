"""Batched pitch worlds, compact records, and observation-only policy rollout."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from aerodrome.composition import WorldParameters
from aerodrome.core.signals import PitchGoal
from aerodrome.runners.batch import BatchedWorld
from aerodrome.runners.episodes import EpisodeRunner, Task, Policy, Outcome
from world_pitch import build_experiment


def build_batch(size=64):
    world, scenario, _ = build_experiment()
    p = scenario.parameters.entities[0]
    # Only the real body's damping varies. The estimator keeps its nominal model.
    axes = jax.tree.map(lambda _: None, p)
    axes = axes._replace(true_model=axes.true_model._replace(damping_s=0))
    parameters = scenario.parameters._replace(entities=(p._replace(
        true_model=p.true_model._replace(damping_s=jnp.linspace(0.5, 1.1, size))),))

    def sample(key, template):
        initial = template["aircraft"]
        truth = initial.truth._replace(pitch_rad=initial.truth.pitch_rad + 0.01*jax.random.normal(key))
        # Preserve the separately specified navigation belief.
        return {"aircraft": initial._replace(truth=truth)}

    batch = BatchedWorld(world, parameter_axes=WorldParameters((axes,), None), initial_sampler=sample)
    state = batch.reset(scenario.seed, np.arange(size), scenario.initial_conditions)
    return batch, state, parameters, scenario.initial_conditions


def main():
    jax.config.update("jax_enable_x64", True)
    batch, state, parameters, initial = build_batch()
    size, steps = len(state.world_id), 240
    targets = jnp.linspace(0.08, 0.12, size)
    actions = (PitchGoal(targets),)
    batch.validate(state, actions, parameters)
    # Projection runs inside scan; avoid retaining every physics substep/state.
    def project(world, trace):
        aircraft = world.entities[0]
        return aircraft.truth.pitch_rad, aircraft.navigation_prior.mean.pitch_rad
    final, compact = jax.jit(lambda s, a, p: batch.rollout_constant(
        s, a, p, steps=steps, record=project))(state, actions, parameters)

    def observe(world, target):
        belief = world.entities[0].navigation_prior.mean
        return jnp.stack((belief.pitch_rad, belief.pitch_rate_rad_s, target))
    def evaluate(obs, action, following, target):
        error = following[0] - target
        return Outcome(-error**2, jnp.abs(following[0]) > 0.5)
    runner = EpisodeRunner(batch, Task(observe, evaluate, 100, parameter_axes=0))
    # A fixed guidance baseline; replace this pure function with a learned policy.
    # It supplies pitch goals to the existing navigation/guidance/PID assembly.
    policy = Policy(lambda obs, p, key: (),
                    lambda memory, obs, p, key: (memory, (PitchGoal(obs[2]),)))
    carry = runner.start_policy(runner.initialize(state, targets), (), policy=policy)
    ending, transitions = jax.jit(lambda c, p, conditions, goals: runner.rollout_policy(
        c, (), p, conditions, goals, policy=policy, steps=steps))(carry, parameters, initial, targets)
    final, compact, ending, transitions = jax.device_get((final, compact, ending, transitions))
    summary = {
        "model": "illustrative two-state pitch; not F16", "device": str(jax.devices()[0]),
        "batch_size": size, "decision_steps": steps, "ticks_per_step": batch.world.spec.ticks_per_step,
        "compact_shape": list(compact[0].shape), "observation_shape": list(transitions.observation.shape),
        "final_physics_tick": int(final.world.tick[0]),
        "terminated": int(np.sum(transitions.terminated)), "truncated": int(np.sum(transitions.truncated)),
        "episode_ids": np.unique(ending.environment.batch.episode_id).tolist(),
        "mean_reward": float(np.mean(transitions.reward)),
    }
    folder = Path("artifacts/batch_rollout")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(folder / "trajectory.npz", truth=compact[0], estimate=compact[1],
             observation=transitions.observation, next_observation=transitions.next_observation,
             reward=transitions.reward, terminated=transitions.terminated, truncated=transitions.truncated,
             world_id=transitions.world_id, episode_id=transitions.episode_id)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
