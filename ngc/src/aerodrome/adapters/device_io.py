"""Bounded host I/O channels and graph adapters; no device calls inside JAX."""
from threading import Lock
from time import perf_counter
import math
import numpy as np
import jax
from aerodrome.adapters.external import Port
from aerodrome.composition.module_graph import Module, ModuleValue


def _copy_values(ports, values):
    if set(values)!={p.name for p in ports}:
        raise ValueError("I/O values must match declared ports")
    result = {}
    for p in ports:
        a = np.asarray(values[p.name])
        if a.shape!=p.shape or a.dtype!=np.dtype(p.dtype):
            raise ValueError(f"{p.name}: I/O shape/dtype mismatch")
        if a.dtype.kind not in "biuf" or not np.all(np.isfinite(a)):
            raise ValueError("I/O payload must contain finite numeric values")
        # Immutable backing bytes: consumers cannot re-enable write access.
        result[p.name] = np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)
    return result


class InputChannel:
    """Thread-safe latest state; sequence IDs support independent readers.

    Timestamps are receipt times in the supplied LOCAL monotonic clock, not
    exposure times or simulation time. Age/staleness uses the same clock.
    """
    def __init__(self,ports, *, clock=perf_counter):
        self.ports = tuple(ports)
        names = [p.name for p in self.ports]
        if not names or len(set(names))!=len(names) or set(names)&{"io_valid","io_fresh","io_age_s","io_sequence"}:
            raise ValueError("empty, duplicate or reserved input port name")
        self._clock,self._lock = clock,Lock()
        self._values = {p.name:np.zeros(p.shape,p.dtype) for p in self.ports}
        self._sequence,self._time,self._connected,self._closed = 0,0.,False,False
        self.error = None

    def publish(self,values):
        values = _copy_values(self.ports,values)
        with self._lock:
            if self._closed:
                raise RuntimeError("input channel closed")
            if self._sequence>=np.iinfo(np.int32).max:
                raise OverflowError("input sequence exhausted; create a new channel")
            self._values,self._time = values,self._clock()
            self._sequence += 1
            self._connected,self.error = True,None

    def disconnect(self,reason="disconnected"):
        with self._lock:
            self._connected,self.error = False,str(reason)

    def read(self, *, previous_sequence=0,max_age_s=.5):
        if not math.isfinite(max_age_s) or max_age_s<0:
            raise ValueError("max age must be finite nonnegative")
        with self._lock:
            age = max(0.,self._clock()-self._time) if self._sequence else 0.
            valid = self._connected and not self._closed and age<=max_age_s
            values = {k:v.copy() for k,v in self._values.items()}
            return dict(values,io_valid=np.asarray(valid,np.bool_),
                        io_fresh=np.asarray(valid and self._sequence!=int(previous_sequence),np.bool_),
                        io_age_s=np.asarray(min(age,np.finfo(np.float32).max),np.float32),
                        io_sequence=np.asarray(self._sequence,np.int32))

    @property
    def output_ports(self):
        return self.ports+(Port("io_valid","1",(),"scalar",dtype="bool"),
                           Port("io_fresh","1",(),"scalar",dtype="bool"),
                           Port("io_age_s","s",(),"wall_clock",dtype="float32"),
                           Port("io_sequence","1",(),"scalar",dtype="int32"))

    def close(self):
        with self._lock:
            self._closed,self._connected = True,False

    def module(self,name, *, every=1,max_age_s=.5):
        """Live host graph source. Compiler executes this only in host regions."""
        def sample(state,inputs,p,context):
            values = self.read(previous_sequence=state,max_age_s=max_age_s)
            return ModuleValue(values["io_sequence"],values)
        return Module(name,(),self.output_ports,sample,backend="host",every=every,
                      equation="host latest input snapshot; receipt-clock freshness")

    def initial_value(self):
        return ModuleValue(np.asarray(0,np.int32),self.read())


def input_module(name,ports, *, every=1):
    """Pure JAX pass-through for recorded or host-snapshotted graph inputs.

    InputBindings connect the supplied port names. Initialize outputs using
    channel.read(). This path can run inside GraphAssembly/World/native scan.
    """
    ports = tuple(ports)
    with_metadata = {"io_sequence","io_fresh","io_valid"}.issubset({p.name for p in ports})
    def sample(state,inputs,p,context):
        outputs = dict(inputs)
        if with_metadata:
            outputs["io_fresh"] = inputs["io_valid"] & (inputs["io_sequence"]!=state)
            state = inputs["io_sequence"]
        return ModuleValue(state,outputs)
    return Module(name,ports,ports,sample,every=every,
                  equation="explicit external input sample; no host I/O")


class OutputChannel:
    """Bounded latest output command, not an actuator or event log.

    Device/UI thread takes and applies commands; graph workers only enqueue.
    Use an acknowledged ordered transport for commands that must never drop.
    """
    def __init__(self,ports):
        self.ports = tuple(ports)
        if not self.ports or len({p.name for p in self.ports})!=len(self.ports):
            raise ValueError("output ports must be nonempty and unique")
        self._lock,self._latest,self._closed = Lock(),None,False
        self.published,self.dropped = 0,0

    def publish(self,values, *, tick,time_s):
        values = _copy_values(self.ports,jax.device_get(values))
        with self._lock:
            if self._closed:
                raise RuntimeError("output channel closed")
            self.dropped += self._latest is not None
            self.published += 1
            self._latest = (int(tick),float(time_s),values)

    def take(self):
        with self._lock:
            value,self._latest = self._latest,None
            return value

    def close(self):
        with self._lock:
            self._closed,self._latest = True,None

    def module(self,name, *, every=1):
        ack = Port("queued","1",(),"scalar",dtype="bool")
        def submit(state,inputs,p,context):
            self.publish(inputs,tick=context.tick,time_s=context.time_s)
            return ModuleValue(state,{"queued":np.asarray(True,np.bool_)})
        return Module(name,self.ports,(ack,),submit,backend="host",every=every,
                      equation="enqueue latest output; queued does not mean applied")
