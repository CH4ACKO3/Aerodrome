"""Generic discrete sensor: bias + random walk + noise, quantization, loss, delay."""
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import NamedTuple, Any
import numpy as np
import jax
import jax.numpy as jnp
from aerodrome.adapters.external import Port
from aerodrome.composition.module_graph import Module, ModuleValue


class SensorParameters(NamedTuple):
    noise_std: Any
    bias: Any
    drift_std_per_sqrt_s: Any
    dropout_probability: Any
    resolution: Any
    lower: Any
    upper: Any

    def validate(self, shape=()):
        """Host-side configuration validation before JIT; loss is per packet."""
        for name in ("noise_std", "drift_std_per_sqrt_s", "resolution"):
            value = np.asarray(getattr(self, name))
            if np.any(~np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        probability = np.asarray(self.dropout_probability)
        if probability.shape != () or not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("dropout_probability must be a scalar in [0,1]")
        if np.any(~np.isfinite(self.bias)) or np.any(np.isnan(self.lower)) or np.any(np.isnan(self.upper)) or np.any(np.asarray(self.lower) > self.upper):
            raise ValueError("bias or bounds invalid")
        for value in self:
            if np.broadcast_shapes(np.shape(value), shape) != shape:
                raise ValueError("sensor parameters must broadcast to signal shape")


class SensorState(NamedTuple):
    key_data: Any
    drift: Any
    buffer: Any
    ticks: Any
    pending: Any
    value: Any
    sample_tick: Any
    valid: Any


@dataclass(frozen=True)
class SampledSensor:
    id: str
    signal: Port
    every: int = 1
    delay_ticks: int = 0

    def __post_init__(self):
        if type(self.every) is not int or self.every < 1 or type(self.delay_ticks) is not int or self.delay_ticks < 0:
            raise ValueError("sensor period must be positive and delay nonnegative integers")
        if not jnp.issubdtype(jnp.dtype(self.signal.dtype), jnp.floating):
            raise ValueError("sensor signal must have a floating dtype")

    def initialize(self, key):
        if jax.random.key_data(key).shape != (2,):
            raise ValueError("sensor requires a scalar threefry2x32 key")
        key = jax.random.fold_in(key, int.from_bytes(sha256(self.id.encode()).digest()[:4], "little"))
        zero = jnp.zeros(self.signal.shape, self.signal.dtype)
        n = self.delay_ticks + 1
        state = SensorState(jax.random.key_data(key), zero, jnp.zeros((n,)+zero.shape, zero.dtype),
                            jnp.full((n,), -1, jnp.int32), jnp.zeros((n,), bool),
                            zero, jnp.asarray(-1, jnp.int32), jnp.asarray(False))
        return ModuleValue(state, self._outputs(state, jnp.asarray(False)))

    @staticmethod
    def _outputs(state, fresh):
        return dict(value=state.value, sample_tick=state.sample_tick, valid=state.valid, fresh=fresh)

    def step(self, state, inputs, p, context):
        dtype = state.value.dtype
        tick = context.tick
        due = tick % self.every == 0
        key = jax.random.fold_in(jax.random.wrap_key_data(state.key_data, impl="threefry2x32"), tick)
        noise_key, drift_key, loss_key = jax.random.split(key, 3)
        # Drift is a per-channel random walk updated at sampling instants.
        drift = state.drift + jnp.asarray(p.drift_std_per_sqrt_s, dtype)*jnp.sqrt(
            jnp.asarray(self.every*context.physics_dt_s, dtype))*jax.random.normal(drift_key, state.value.shape, dtype)
        drift = jnp.where(due, drift, state.drift)
        raw = inputs["truth"] + jnp.asarray(p.bias, dtype) + drift + jnp.asarray(p.noise_std, dtype)*jax.random.normal(
            noise_key, state.value.shape, dtype)
        resolution = jnp.asarray(p.resolution, dtype)
        quantized = jnp.round(raw/jnp.where(resolution > 0, resolution, 1))*resolution
        raw = jnp.clip(jnp.where(resolution > 0, quantized, raw), p.lower, p.upper).astype(dtype)
        accepted = due & (jax.random.uniform(loss_key) >= p.dropout_probability)
        index = tick % (self.delay_ticks+1)
        buffer = state.buffer.at[index].set(raw)
        ticks = state.ticks.at[index].set(tick)
        pending = state.pending.at[index].set(accepted)
        delivery = (index+1) % (self.delay_ticks+1)
        fresh = pending[delivery]
        following = SensorState(state.key_data, drift, buffer, ticks, pending,
                                jnp.where(fresh, buffer[delivery], state.value),
                                jnp.where(fresh, ticks[delivery], state.sample_tick), state.valid | fresh)
        return ModuleValue(following, self._outputs(following, fresh))

    def module(self):
        # Run every physics tick to deliver delayed samples and clear fresh pulses.
        # Sample scheduling belongs inside this module, not Module.every.
        return Module(self.id, (replace(self.signal, name="truth"),),
                      (replace(self.signal, name="value"),
                       Port("sample_tick", "1", (), "clock", dtype="int32", quantity="sample_tick"),
                       Port("valid", "1", (), "logical", dtype="bool", quantity="valid"),
                       Port("fresh", "1", (), "logical", dtype="bool", quantity="fresh")),
                      self.step, equation="y=clip(quantize(h(x)+bias+drift+sigma*epsilon)); sampled/delayed")
