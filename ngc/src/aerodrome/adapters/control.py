"""Host-only python-control/SciPy bridge; exports explicit JAX numerical matrices."""
from dataclasses import dataclass
import math
import numpy as np
import jax
import jax.numpy as jnp
from scipy.signal import cont2discrete, tf2ss
from aerodrome.models.linear import LinearParameters, step, output
from aerodrome.composition.module_graph import Module, ModuleValue


def _positive_dt(value):
    if isinstance(value, (bool,np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError("sampling time must be an explicit finite positive number")
    return float(value)


def _matrices(A,B,C,D):
    arrays = tuple(np.asarray(x) for x in (A,B,C,D))
    if any(x.ndim != 2 or x.dtype.kind not in "fiu" or not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("A/B/C/D must be finite real rank-2 matrices")
    a,b,c,d = arrays
    n = a.shape[0]
    if a.shape != (n,n) or b.shape[0] != n or c.shape[1] != n or d.shape != (c.shape[0],b.shape[1]) or not b.shape[1] or not c.shape[0]:
        raise ValueError("incompatible state/input/output matrix dimensions")
    return arrays


@dataclass(frozen=True, eq=False)
class DiscreteLinearModel:
    parameters: LinearParameters
    dt: float
    input_labels: tuple = ()
    output_labels: tuple = ()
    state_labels: tuple = ()
    conversion_notes: tuple = ()

    def initialize(self, state=None, inputs=None, *, parameters=None):
        p = self.parameters if parameters is None else parameters
        dtype = p.A.dtype
        state = jnp.zeros((p.A.shape[0],),dtype) if state is None else jnp.asarray(state,dtype)
        inputs = jnp.zeros((p.B.shape[1],),dtype) if inputs is None else jnp.asarray(inputs,dtype)
        if state.shape != (p.A.shape[0],) or inputs.shape != (p.B.shape[1],):
            raise ValueError("initial state/input must be vectors with the declared dimensions")
        return state, output(state,inputs,p)

    def module(self, name, input_port, output_port, *, every=1):
        """Atomic graph module, with conservative same-tick dependency semantics.

        Use returned numerical parameters separately in graph/World parameters.
        A/B/C/D are not captured as constants by this module's step function.
        """
        n,nu,ny = self.parameters.A.shape[0],self.parameters.B.shape[1],self.parameters.C.shape[0]
        dtype,dt = np.dtype(self.parameters.A.dtype),self.dt
        if input_port.shape != (nu,) or output_port.shape != (ny,):
            raise ValueError("linear module ports must be vectors [nu] and [ny], including SISO")
        if np.dtype(input_port.dtype) != dtype or np.dtype(output_port.dtype) != dtype:
            raise ValueError("linear module port dtype must match matrix dtype")
        def advance(state, inputs, p, context):
            # ModuleGraph constructs this period from static scheduling metadata.
            if not math.isclose(context.sample_period_s,dt,rel_tol=1e-12,abs_tol=0):
                raise ValueError("module sample period differs from the discretized model dt")
            if state.shape != (n,) or tuple(x.shape for x in p) != ((n,n),(n,nu),(ny,n),(ny,nu)):
                raise ValueError("linear runtime state/parameter dimensions differ from module")
            if state.dtype != dtype or any(x.dtype != dtype for x in p):
                raise ValueError("linear runtime state/parameter dtype differs from module")
            following,y = step(state,inputs[input_port.name],p)
            return ModuleValue(following,{output_port.name:y})
        return Module(name,(input_port,),(output_port,),advance,every=every,
                      equation="y[k]=C*x[k]+D*u[k]; x[k+1]=A*x[k]+B*u[k]")


def from_matrices(A,B,C,D, *, dt=None, sample_time=None, method="zoh", dtype="float64"):
    """Build once on the host. dt=None: continuous; dt>0: already discrete.

    Continuous matrices require sample_time; only ZOH and bilinear/Tustin are
    exposed here. Gradient through host discretization is intentionally absent.
    """
    arrays = _matrices(A,B,C,D)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"),np.dtype("float64")):
        raise ValueError("linear model dtype must be float32 or float64")
    if dtype == np.dtype("float64") and not jax.config.x64_enabled:
        raise ValueError("enable JAX x64 explicitly or request dtype='float32'")
    if dt is None:
        period = _positive_dt(sample_time)
        if method not in ("zoh","bilinear","tustin"):
            raise ValueError("supported discretizations: zoh, bilinear/tustin")
        arrays = cont2discrete(tuple(np.asarray(x,np.float64) for x in arrays),period,
                              method="bilinear" if method == "tustin" else method)[:4]
    else:
        period = _positive_dt(dt)
        if sample_time is not None and not math.isclose(_positive_dt(sample_time),period,rel_tol=1e-12,abs_tol=0):
            raise ValueError("already-discrete model cannot be resampled implicitly")
    arrays = tuple(np.asarray(x,dtype) for x in arrays)
    _matrices(*arrays)  # Catch nonfinite discretization/cast results before dispatch.
    return DiscreteLinearModel(LinearParameters(*map(jnp.asarray,arrays)),period)


def _mimo_realization(system, max_states):
    """Channel-wise exact rational realization; generally NOT minimal."""
    import control as ct
    if type(max_states) is not int or max_states < 0:
        raise ValueError("max_states must be a nonnegative integer")
    channels, count = [], 0
    for row in range(system.noutputs):
        for col in range(system.ninputs):
            num = np.trim_zeros(np.asarray(system.num[row][col]),"f")
            den = np.trim_zeros(np.asarray(system.den[row][col]),"f")
            if not len(den) or not np.all(np.isfinite(den)) or not np.all(np.isfinite(num)):
                raise ValueError("invalid transfer-function coefficients")
            if len(num) > len(den):
                raise ValueError("non-proper transfer-function channel requires a different model class")
            order = len(den)-1 if len(num) else 0
            count += order
            if count > max_states:
                raise ValueError(f"channel realization exceeds max_states={max_states}; supply a compact SS realization")
            channels.append((row,col,num,den,order))
    A,B,C,D = np.zeros((count,count)),np.zeros((count,system.ninputs)),np.zeros((system.noutputs,count)),np.zeros((system.noutputs,system.ninputs))
    offset,labels = 0,[]
    for row,col,num,den,order in channels:
        if not len(num):
            continue
        if order == 0:
            D[row,col] = num[0]/den[0]
            continue
        a,b,c,d = tf2ss(num,den)
        slot = slice(offset,offset+order)
        A[slot,slot],B[slot,col],C[row,slot],D[row,col] = a,b[:,0],c[0],d[0,0]
        labels.extend(f"y{row}_u{col}_x{k}" for k in range(order))
        offset += order
    return ct.ss(A,B,C,D,dt=system.dt,inputs=system.input_labels,outputs=system.output_labels,states=labels)


def from_control(system, *, sample_time=None, method="zoh", dtype="float64", max_states=256):
    """Import real proper TF/SS, with bounded channel-wise MIMO TF realization."""
    import control as ct  # Optional dependency, deliberately lazy.
    if not isinstance(system,(ct.StateSpace,ct.TransferFunction)):
        raise TypeError("expected python-control StateSpace or TransferFunction")
    if system.dt is None or isinstance(system.dt,(bool,np.bool_)):
        raise ValueError("python-control timebase must be explicit: dt=0 or dt>0")
    notes = ()
    if isinstance(system,ct.TransferFunction):
        if system.issiso():
            system = ct.tf2ss(system,method="scipy")
        else:
            system = _mimo_realization(system,max_states)
            notes = ("Channel-wise MIMO TF realization; exact input/output map, generally nonminimal. No automatic pole cancellation.",)
    model = from_matrices(system.A,system.B,system.C,system.D,
                          dt=None if system.dt == 0 else system.dt,
                          sample_time=sample_time,method=method,dtype=dtype)
    return DiscreteLinearModel(model.parameters,model.dt,tuple(system.input_labels),
                               tuple(system.output_labels),tuple(system.state_labels),notes)


def from_descriptor(A,B,C,D,E, *, dt=None, sample_time=None, method="zoh", dtype="float64", max_condition=1e10):
    """E*x'=A*x+B*u (or E*x_next), only for well-conditioned invertible E.

    Singular descriptor systems need DAE/descriptor algorithms; never use pinv.
    """
    A,B,C,D = _matrices(A,B,C,D)
    E = np.asarray(E)
    if E.shape != A.shape or E.dtype.kind not in "fiu" or not np.all(np.isfinite(E)):
        raise ValueError("E must be a finite real square matrix matching A")
    if not np.isfinite(max_condition) or max_condition < 1:
        raise ValueError("max_condition must be finite and >= 1")
    condition = np.linalg.cond(E) if E.size else 1.
    if not np.isfinite(condition) or condition > max_condition:
        raise ValueError("singular or ill-conditioned E requires a descriptor/DAE solver")
    model = from_matrices(np.linalg.solve(E,A),np.linalg.solve(E,B),C,D,
                          dt=dt,sample_time=sample_time,method=method,dtype=dtype)
    return DiscreteLinearModel(model.parameters,model.dt,conversion_notes=(f"Invertible descriptor E eliminated with solve; cond(E)={condition:g}.",))


def to_control(model):
    """Export the current discrete matrices for host analysis/plots."""
    import control as ct
    kwargs = {name:list(labels) for name,labels in (("inputs",model.input_labels),
              ("outputs",model.output_labels),("states",model.state_labels)) if labels}
    return ct.ss(*jax.device_get(model.parameters),dt=model.dt,**kwargs)
