"""Batch one fixed World topology; share parameters using explicit vmap axes."""
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple
import numpy as np
import jax
import jax.numpy as jnp


class BatchState(NamedTuple):
    world: Any                # WorldState leaves [B, ...], including per-world tick
    world_id: Any             # stable uint32 identities, not positions in a batch
    episode_id: Any           # uint32, increments only on reset
    root_key: Any             # one shared typed JAX key


def select_batch(mask, selected, other):
    """Select complete worlds, including typed PRNG keys; no array compaction."""
    def choose(a, b):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            raw_a, raw_b = jax.random.key_data(a), jax.random.key_data(b)
            shape = mask.shape + (1,) * (raw_a.ndim - mask.ndim)
            return jax.random.wrap_key_data(jnp.where(mask.reshape(shape), raw_a, raw_b),
                                            impl=jax.random.key_impl(a))
        shape = mask.shape + (1,) * (a.ndim - mask.ndim)
        return jnp.where(mask.reshape(shape), a, b)
    return jax.tree.map(choose, selected, other)


def episode_keys(root_key, world_ids, episode_ids, *, stream, ticks=None):
    def key(world_id, episode_id):
        result = jax.random.fold_in(root_key, world_id)
        result = jax.random.fold_in(result, episode_id)
        return jax.random.fold_in(result, stream)
    keys = jax.vmap(key)(world_ids, episode_ids)
    return keys if ticks is None else jax.vmap(jax.random.fold_in)(keys, ticks)


def _identity_initial(key, initial):
    return initial


@dataclass(frozen=True, eq=False)
class BatchedWorld:
    world: Any
    parameter_axes: Any = None  # None: shared; 0 or PyTree of 0/None: mapped leaves
    initial_axes: Any = None
    input_axes: Any = 0
    initial_sampler: Callable = _identity_initial

    def initialize(self, root_key, world_ids, episode_ids, initial_conditions):
        """Pure initialization. IDs are prevalidated uint32 arrays of equal shape."""
        sample_keys = episode_keys(root_key, world_ids, episode_ids, stream=0x53434E52)
        model_keys = episode_keys(root_key, world_ids, episode_ids, stream=0x574F524C)
        def one(sample_key, model_key, template):
            conditions = self.initial_sampler(sample_key, template)
            return self.world.reset_from_key(model_key, conditions)
        states = jax.vmap(one, in_axes=(0, 0, self.initial_axes))(sample_keys, model_keys, initial_conditions)
        return BatchState(states, world_ids, episode_ids, root_key)

    def reset(self, seed, world_ids, initial_conditions):
        """Host-checked entry point. Stable IDs survive reordering/sub-batching."""
        ids = np.asarray(world_ids)
        if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu" or
                np.any(ids < 0) or np.any(ids > np.iinfo(np.uint32).max) or len(np.unique(ids)) != len(ids)):
            raise ValueError("world_ids must be a nonempty vector of unique uint32-range integers")
        ids = jnp.asarray(ids, jnp.uint32)
        return self.initialize(jax.random.key(seed), ids, jnp.zeros_like(ids), initial_conditions)

    def reset_where(self, state, mask, initial_conditions):
        """Reset selected slots to the next episode; other states/keys stay exact."""
        if mask.shape != state.world_id.shape or mask.dtype != jnp.bool_:
            raise ValueError("reset mask must be bool with shape [B]")
        def reset_selected():
            episodes = state.episode_id + mask.astype(jnp.uint32)
            fresh = self.initialize(state.root_key, state.world_id, episodes, initial_conditions)
            return BatchState(select_batch(mask, fresh.world, state.world), state.world_id,
                              episodes, state.root_key)
        return jax.lax.cond(jnp.any(mask), reset_selected, lambda: state)

    def tick(self, state, inputs, parameters):
        following, record = jax.vmap(self.world.tick, in_axes=(0, self.input_axes, self.parameter_axes))(
            state.world, inputs, parameters)
        return state._replace(world=following), record

    def step(self, state, inputs, parameters):
        """B independent decision steps. Full records have axes [B,S,...]."""
        following, record = jax.vmap(self.world.step, in_axes=(0, self.input_axes, self.parameter_axes))(
            state.world, inputs, parameters)
        return state._replace(world=following), record

    @staticmethod
    def _record(state, trace, record):
        if record is None:
            return ()
        if record == "full":
            return trace
        if callable(record):
            # Projection happens inside the compiled numerical path.
            return jax.vmap(record)(state.world, trace)
        raise ValueError("record must be 'full', None, or a per-world projection function")

    def rollout(self, state, input_sequences, parameters, *, steps, record="full"):
        """Time-major sequences. Mapped input leaves [T,B,...], shared [T,...]."""
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive static integer")
        def body(current, inputs):
            following, trace = self.step(current, inputs, parameters)
            return following, self._record(following, trace, record)
        return jax.lax.scan(body, state, input_sequences, length=steps)

    def rollout_constant(self, state, inputs, parameters, *, steps, record="full"):
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive static integer")
        def body(current, _):
            following, trace = self.step(current, inputs, parameters)
            return following, self._record(following, trace, record)
        return jax.lax.scan(body, state, None, length=steps)

    def validate(self, state, inputs, parameters):
        """Check one batched step without running a trajectory."""
        if state.world_id.ndim != 1 or state.episode_id.shape != state.world_id.shape:
            raise ValueError("world_id and episode_id must have shape [B]")
        if state.world.tick.shape != state.world_id.shape:
            raise ValueError("batched world tick must have shape [B]")
        following, _ = jax.eval_shape(self.step, state, inputs, parameters)
        def signature(tree):
            return jax.tree.structure(tree), [(x.shape, x.dtype) for x in jax.tree.leaves(tree)]
        if signature(following) != signature(state):
            raise ValueError("batch step must preserve all state shapes and dtypes")
