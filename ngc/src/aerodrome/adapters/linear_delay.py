"""Exact per-input discrete delays with a ring buffer, not dense state augmentation."""
from dataclasses import dataclass
from typing import NamedTuple, Any
import numpy as np
import jax.numpy as jnp
from aerodrome.composition.module_graph import Module, ModuleValue


class DelayState(NamedTuple):
    plant: Any
    history: Any
    cursor: Any


@dataclass(frozen=True,eq=False)
class InputDelayedModel:
    base: Any
    steps: tuple

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def dt(self):
        return self.base.dt

    def initialize(self,state=None,inputs=None,*,parameters=None,history=None):
        p = self.parameters if parameters is None else parameters
        ninputs = p.B.shape[1]
        length = max(self.steps)
        history = jnp.zeros((length,ninputs),p.A.dtype) if history is None else jnp.asarray(history,p.A.dtype)
        if history.shape != (length,ninputs):
            raise ValueError("history must be [max_delay,nu], oldest to newest")
        inputs = jnp.zeros(ninputs,p.A.dtype) if inputs is None else jnp.asarray(inputs,p.A.dtype)
        delayed = self._read(history,jnp.int32(0),inputs)
        x,y = self.base.initialize(state,delayed,parameters=p)
        return DelayState(x,history,jnp.int32(0)),y

    def _read(self,history,cursor,inputs):
        length = max(self.steps)
        if not length:
            return inputs
        steps = jnp.asarray(self.steps,jnp.int32)
        stored = history[(cursor-steps)%length,jnp.arange(len(self.steps))]
        return jnp.where(steps == 0,inputs,stored)

    def module(self,name,input_port,output_port,*,every=1):
        base = self.base.module(name,input_port,output_port,every=every)
        length = max(self.steps)
        def advance(state,inputs,p,context):
            current = inputs[input_port.name]
            delayed = self._read(state.history,state.cursor,current)
            result = base.step(state.plant,{input_port.name:delayed},p,context)
            history = state.history.at[state.cursor].set(current) if length else state.history
            cursor = (state.cursor+1)%length if length else state.cursor
            return ModuleValue(DelayState(result.state,history,cursor),result.outputs)
        return Module(name,base.inputs,base.outputs,advance,every=every,
                      equation="v_i[k]=u_i[k-delay_i]; y[k]=Cx[k]+Dv[k]; x[k+1]=Ax[k]+Bv[k]")


def with_input_delay(model,steps,*,max_buffer_values=1_000_000):
    """steps counts model updates, not physics ticks; fractional values are refused."""
    values = np.asarray(steps)
    if values.shape == ():
        values = np.full(model.parameters.B.shape[1],values)
    if values.shape != (model.parameters.B.shape[1],) or values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("delay must be nonnegative integer steps, scalar or one per input")
    if type(max_buffer_values) is not int or max_buffer_values < 0 or int(np.max(values))*len(values) > max_buffer_values:
        raise ValueError("input delay exceeds max_buffer_values; choose an explicit memory budget")
    return InputDelayedModel(model,tuple(map(int,values)))
