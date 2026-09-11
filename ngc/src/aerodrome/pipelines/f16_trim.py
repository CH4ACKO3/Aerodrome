"""Host trim/linearization/LQR design for a specified level-flight operating point."""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares
from aerodrome.models.f16_longitudinal import rhs,Airframe
from aerodrome.adapters.control import from_control


def design(tables,*,speed=150.,height=3000.,dt=.02,airframe=Airframe()):
    import control as ct
    def unpack(z):
        alpha,elevator,thrust_scaled = z
        return jnp.array([speed,alpha,0.,alpha,height]),jnp.array([elevator,thrust_scaled*10000])
    def residual(z):
        x,u = unpack(z)
        return rhs(x,u,tables,airframe)[:3]*jnp.array([1.,speed,10.])
    evaluate = jax.jit(residual)
    derivative = jax.jit(jax.jacfwd(residual))
    result = least_squares(lambda z:np.asarray(evaluate(z)),[.05,-.03,1.],
                           jac=lambda z:np.asarray(derivative(z)),
                           bounds=([-np.deg2rad(5),-np.deg2rad(25),0.],
                                   [np.deg2rad(15),np.deg2rad(25),8.45]),
                           xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=200)
    x,u = unpack(result.x)
    residual_value = np.asarray(rhs(x,u,tables,airframe))
    if not result.success or np.max(np.abs(residual_value)) > 1e-8:
        raise RuntimeError(f"level-flight trim failed: {result.message}; residual={residual_value}")
    A,B = jax.jit(jax.jacfwd(rhs,argnums=(0,1)))(x,u,tables,airframe)
    A,B = np.asarray(A),np.asarray(B)
    sx = np.array([5.,np.deg2rad(2),np.deg2rad(2),np.deg2rad(2),20.])
    su = np.array([np.deg2rad(5),10000.])
    normalized = from_control(ct.ss(A*sx[None,:]/sx[:,None],B*su[None,:]/sx[:,None],np.eye(5),np.zeros((5,2)),dt=0),sample_time=dt)
    F,G = map(np.asarray,normalized.parameters[:2])
    K,P,poles = ct.dlqr(F,G,np.diag([2.,1.,.2,.5,2.]),np.diag([25.,2.]),method="scipy")
    gain = su[:,None]*np.asarray(K)/sx[None,:]
    if np.max(np.abs(poles)) >= 1:
        raise RuntimeError("discrete closed loop is not asymptotically stable")
    return x,u,jnp.asarray(gain),dict(A=A,B=B,closed_loop_poles=poles,
                                      trim_residual=residual_value,normalized_riccati=P,
                                      normalized_K=K,normalized_F=F,normalized_G=G)
