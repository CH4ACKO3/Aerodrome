"""GPU-compatible episodic batch lifecycle; task and policy stay outside physics."""
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple
import jax
import jax.numpy as jnp
from .batch import episode_keys, select_batch


class Outcome(NamedTuple):
    reward: Any
    terminated: Any


@dataclass(frozen=True)
class Task:
    observe: Callable  # (single WorldState, task_parameters) -> allowed observation
    evaluate: Callable # (observation, action, next_observation, task_parameters) -> Outcome
    max_episode_steps: int
    parameter_axes: Any = None
    reward_dtype: str = "float32"

    def __post_init__(self):
        if type(self.max_episode_steps) is not int or self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        if not jnp.issubdtype(jnp.dtype(self.reward_dtype), jnp.floating):
            raise ValueError("reward_dtype must be floating point")


@dataclass(frozen=True)
class Policy:
    initialize: Callable  # (observation, parameters, key) -> per-world recurrent state, or ()
    step: Callable        # (policy_state, observation, parameters, key) -> (state, action)
    parameter_axes: Any = None


class EpisodeState(NamedTuple):
    batch: Any
    observation: Any
    episode_steps: Any
    episode_return: Any


class Transition(NamedTuple):
    world_id: Any
    episode_id: Any
    episode_step: Any
    observation: Any
    action: Any
    reward: Any
    next_observation: Any  # terminal observation BEFORE any reset
    terminated: Any
    truncated: Any
    episode_return: Any


class PolicyState(NamedTuple):
    environment: EpisodeState
    policy: Any


class EpisodeRunner:
    """Always auto-reset completed slots. Termination is checked at decision boundaries.

    Task.observe is the sole policy visibility boundary. Policies/evaluate receive
    observations, never a WorldState or hidden physical parameters.
    """
    def __init__(self, batch, task):
        if batch.input_axes != 0:
            raise ValueError("episodic actions must be mapped on axis 0 for every leaf")
        self.batch, self.task = batch, task

    def _observe(self, state, parameters):
        return jax.vmap(self.task.observe, in_axes=(0, self.task.parameter_axes))(state.world, parameters)

    def initialize(self, batch_state, task_parameters):
        return EpisodeState(batch_state, self._observe(batch_state, task_parameters),
                            jnp.zeros_like(batch_state.world_id, dtype=jnp.int32),
                            jnp.zeros_like(batch_state.world_id, dtype=self.task.reward_dtype))

    def step(self, state, actions, parameters, initial_conditions, task_parameters):
        terminal, _ = self.batch.step(state.batch, actions, parameters)
        terminal_obs = self._observe(terminal, task_parameters)
        outcome = jax.vmap(self.task.evaluate, in_axes=(0, 0, 0, self.task.parameter_axes))(
            state.observation, actions, terminal_obs, task_parameters)
        shape = state.batch.world_id.shape
        if outcome.reward.shape != shape or outcome.terminated.shape != shape or outcome.terminated.dtype != jnp.bool_:
            raise ValueError("task must return scalar reward and scalar bool terminated per world")
        reward = outcome.reward.astype(self.task.reward_dtype)
        steps = state.episode_steps + 1
        truncated = (steps >= self.task.max_episode_steps) & ~outcome.terminated
        done = outcome.terminated | truncated
        returns = state.episode_return + reward
        transition = Transition(state.batch.world_id, state.batch.episode_id, steps,
                                state.observation, actions, reward, terminal_obs,
                                outcome.terminated, truncated, returns)
        following = self.batch.reset_where(terminal, done, initial_conditions)
        observation = select_batch(done, self._observe(following, task_parameters), terminal_obs)
        return EpisodeState(following, observation, jnp.where(done, 0, steps),
                            jnp.where(done, jnp.zeros_like(returns), returns)), transition

    def rollout(self, state, action_sequences, parameters, initial_conditions, task_parameters, *, steps):
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive static integer")
        return jax.lax.scan(lambda current, actions: self.step(
            current, actions, parameters, initial_conditions, task_parameters),
            state, action_sequences, length=steps)

    @staticmethod
    def _policy_keys(state, *, initialize=False):
        return episode_keys(state.batch.root_key, state.batch.world_id, state.batch.episode_id,
                            stream=0x50494E49 if initialize else 0x504F4C59,
                            ticks=None if initialize else state.episode_steps)

    def start_policy(self, state, policy_parameters, *, policy):
        recurrent = jax.vmap(policy.initialize, in_axes=(0, policy.parameter_axes, 0))(
            state.observation, policy_parameters, self._policy_keys(state, initialize=True))
        return PolicyState(state, recurrent)

    def rollout_policy(self, state, policy_parameters, parameters, initial_conditions, task_parameters,
                       *, policy, steps):
        """Return recurrent carry + compact [T,B,...] transitions; no host callbacks."""
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive static integer")
        def body(current, _):
            recurrent, actions = jax.vmap(policy.step, in_axes=(0, 0, policy.parameter_axes, 0))(
                current.policy, current.environment.observation, policy_parameters,
                self._policy_keys(current.environment))
            following, transition = self.step(current.environment, actions, parameters,
                                               initial_conditions, task_parameters)
            fresh = self.start_policy(following, policy_parameters, policy=policy).policy
            recurrent = select_batch(transition.terminated | transition.truncated, fresh, recurrent)
            return PolicyState(following, recurrent), transition
        return jax.lax.scan(body, state, None, length=steps)
