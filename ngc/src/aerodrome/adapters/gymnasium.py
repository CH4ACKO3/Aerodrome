"""Gymnasium host interface over a native World, independent of RL algorithms."""
from typing import NamedTuple,Any
import jax
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import ResetNeeded


class ResetSpec(NamedTuple):
    initial_conditions: Any
    parameters: Any
    task_parameters: Any = ()


def _numeric_space(space):
    if isinstance(space,spaces.Dict):
        return all(_numeric_space(s) for s in space.spaces.values())
    if isinstance(space,spaces.Tuple):
        return all(_numeric_space(s) for s in space.spaces)
    return isinstance(space,(spaces.Box,spaces.Discrete,spaces.MultiDiscrete,spaces.MultiBinary))


def _observation(value,space):
    if isinstance(space,spaces.Dict):
        if set(value)!=set(space.spaces):
            raise ValueError("observation keys differ from observation_space")
        return {k:_observation(value[k],s) for k,s in space.spaces.items()}
    if isinstance(space,spaces.Tuple):
        if len(value)!=len(space.spaces):
            raise ValueError("observation tuple length differs from observation_space")
        return tuple(_observation(v,s) for v,s in zip(value,space.spaces,strict=True))
    array = np.array(value,copy=True)
    return array[()] if isinstance(space,spaces.Discrete) and array.shape==() else array


class WorldEnv(gym.Env):
    """One agent controls one World (which may contain multiple entities).

    Reuses runners.episodes.Task. Task.evaluate sees the PUBLIC action; the
    action_to_inputs(action, state, parameters) callback maps it to World inputs.
    Task/mapper must be pure JAX functions. reset_factory(rng, options) is host
    code returning ResetSpec; it may randomize initial conditions and parameters.
    Time limits are decision steps; no implicit autoreset or action clipping.
    """
    metadata = {"render_modes":["ansi"]}

    def __init__(self,world, *, initial_conditions,parameters,task,action_space,
                 observation_space,task_parameters=(),action_to_inputs=None,
                 reset_factory=None,jit=True,render_mode=None):
        if render_mode not in (None,"ansi"):
            raise ValueError("WorldEnv currently supports render_mode=None or ansi")
        if not _numeric_space(action_space) or not _numeric_space(observation_space):
            raise ValueError("WorldEnv supports numeric Box/Discrete/MultiDiscrete/MultiBinary/Dict/Tuple spaces")
        self.world,self.task = world,task
        self.action_space,self.observation_space = action_space,observation_space
        self.render_mode = render_mode
        self._defaults = ResetSpec(initial_conditions,parameters,task_parameters)
        self._reset_factory = reset_factory
        mapper = action_to_inputs or (lambda action,state,parameters:action)

        def transition(state,observation,action,parameters,task_parameters):
            following,_ = world.step(state,mapper(action,state,parameters),parameters)
            next_observation = task.observe(following,task_parameters)
            outcome = task.evaluate(observation,action,next_observation,task_parameters)
            reward,terminated = jnp.asarray(outcome.reward),jnp.asarray(outcome.terminated)
            if reward.shape!=() or reward.dtype.kind!="f" or terminated.shape!=() or terminated.dtype!=jnp.bool_:
                raise ValueError("task must return scalar floating reward and scalar bool terminated")
            return following,next_observation,reward.astype(task.reward_dtype),terminated

        # Also usable directly by native JAX callers; no NumPy/Gym operations here.
        self.transition = jax.jit(transition) if jit else transition
        self.state = None
        self._needs_reset,self._closed = True,False

    def _info(self):
        tick = int(jax.device_get(self.state.tick))
        return {"tick":tick,"time_s":tick*self.world.spec.schedule.physics_dt_s,
                "episode_steps":self._steps,"episode_return":self._return}

    def _checked_observation(self,value):
        result = _observation(value,self.observation_space)
        if not self.observation_space.contains(result):
            raise ValueError("task observation violates observation_space")
        if any(not np.all(np.isfinite(x)) for x in jax.tree.leaves(result)):
            raise FloatingPointError("nonfinite observation")
        return result

    def reset(self, *, seed=None,options=None):
        if self._closed:
            raise RuntimeError("environment is closed")
        super().reset(seed=seed)
        self._needs_reset = True
        options = {} if options is None else dict(options)
        if options and self._reset_factory is None:
            raise ValueError("reset options require a reset_factory")
        spec = self._reset_factory(self.np_random,options) if self._reset_factory else self._defaults
        if not isinstance(spec,ResetSpec):
            raise TypeError("reset_factory must return ResetSpec")
        self.parameters,self.task_parameters = spec.parameters,spec.task_parameters
        world_seed = int(self.np_random.integers(0,2**32,dtype=np.uint64))
        state = self.world.reset(world_seed,spec.initial_conditions)
        observation = self.task.observe(state,self.task_parameters)
        result = self._checked_observation(jax.device_get(observation))
        self.state,self._observation = state,observation
        self._steps,self._return,self._needs_reset = 0,0.,False
        return result,self._info()

    def step(self,action):
        if self._closed:
            raise RuntimeError("environment is closed")
        if self._needs_reset:
            raise ResetNeeded("call reset before stepping or after episode completion")
        if not self.action_space.contains(action):
            raise ValueError("action violates action_space; actions are not clipped")
        if any(not np.all(np.isfinite(x)) for x in jax.tree.leaves(action)):
            raise ValueError("action must be finite")
        try:
            following,observation,reward,terminated = self.transition(
                self.state,self._observation,jax.tree.map(jnp.asarray,action),self.parameters,self.task_parameters)
            host_obs,reward,terminated = jax.device_get((observation,reward,terminated))
            result = self._checked_observation(host_obs)
            reward,terminated = float(reward),bool(terminated)
            if not np.isfinite(reward):
                raise FloatingPointError("nonfinite reward")
        except Exception:
            self._needs_reset = True
            raise
        self.state,self._observation = following,observation
        self._steps += 1
        self._return += reward
        truncated = self._steps>=self.task.max_episode_steps and not terminated
        self._needs_reset = terminated or truncated
        return result,reward,terminated,truncated,self._info()

    def render(self):
        if self._closed:
            raise RuntimeError("environment is closed")
        if self.render_mode is None:
            return None
        if self.state is None:
            raise ResetNeeded("reset before rendering")
        info = self._info()
        return f"World tick={info['tick']} time={info['time_s']:.6f}s entities={self.world.entity_ids}"

    def close(self):
        self._closed,self._needs_reset = True,True
        self.state,self._observation = None,None
