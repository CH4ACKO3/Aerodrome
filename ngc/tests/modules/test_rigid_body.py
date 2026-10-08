import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from aerodrome.models.rigid_body import (RigidBody6DoF, BodyLoads, mass_properties,
    euler321_to_quaternion, quaternion_to_euler321, quaternion_to_matrix,
    force_at_point, sum_loads)


def simulate(model, state, parameters, loads, dt=.01, steps=1000, **kwargs):
    def run(s):
        def tick(x, _):
            y = model.step(x, loads, parameters, dt, **kwargs)
            return y, y
        return jax.lax.scan(tick, s, None, length=steps)
    return jax.jit(run)(state)


def test_attitude_conventions_against_scipy():
    angles = np.random.default_rng(8).uniform(-1., 1., (30, 3))
    qs = jax.jit(jax.vmap(euler321_to_quaternion))(jnp.asarray(angles))
    matrices = jax.vmap(quaternion_to_matrix)(qs)
    np.testing.assert_allclose(matrices, Rotation.from_euler("xyz", angles).as_matrix(), atol=1e-14)
    np.testing.assert_allclose(jax.vmap(quaternion_to_euler321)(qs), angles, atol=1e-14)
    np.testing.assert_allclose(jax.vmap(quaternion_to_matrix)(-qs), matrices, atol=1e-14)
    for pitch in [-np.pi/2, np.pi/2]:
        q = euler321_to_quaternion(jnp.array([.4, pitch, -.7]))
        rebuilt = euler321_to_quaternion(quaternion_to_euler321(q))
        np.testing.assert_allclose(quaternion_to_matrix(q), quaternion_to_matrix(rebuilt), atol=1e-14)


@pytest.mark.parametrize("mode", ["quaternion", "euler321"])
def test_constant_acceleration_analytic(mode):
    model = RigidBody6DoF(mode)
    p = mass_properties(2., np.eye(3))
    x = model.initialize(position_ned_m=[2, 3, -4], velocity_body_m_s=[10, 0, 0])
    y, _ = simulate(model, x, p, BodyLoads(jnp.array([4., 0, 0]), jnp.zeros(3)), steps=100)
    a = np.array([2., 0, 9.80665])
    np.testing.assert_allclose(y.position_ned_m, np.array([2,3,-4])+[10,0,0]+.5*a, atol=1e-12)
    np.testing.assert_allclose(y.velocity_body_m_s, np.array([10,0,0])+a, atol=1e-12)


def test_free_rotation_conservation_and_euler_equivalence():
    I = np.array([[2., .1, -.2], [.1, 2.8, .15], [-.2, .15, 3.4]])
    p = mass_properties(4., I, [0,0,0])
    loads = BodyLoads(jnp.zeros(3), jnp.zeros(3))
    quat, euler = RigidBody6DoF(), RigidBody6DoF("euler321")
    init = dict(euler_rad=[.2,-.3,.4], velocity_body_m_s=[4,2,-1], omega_body_rad_s=[.12,.2,-.1])
    x = quat.initialize(**init)
    y, trace = simulate(quat, x, p, loads)
    z, _ = simulate(euler, euler.initialize(**init), p, loads)
    R = jax.vmap(quaternion_to_matrix)(trace.attitude)
    H = jnp.einsum("tij,tj->ti", R, trace.omega_body_rad_s@I.T)
    H0 = quat.rotation_matrix(x)@I@x.omega_body_rad_s
    np.testing.assert_allclose(H, np.broadcast_to(H0,H.shape), atol=2e-11)
    energy = jnp.einsum("ti,ij,tj->t",trace.omega_body_rad_s,I,trace.omega_body_rad_s)/2
    np.testing.assert_allclose(energy, x.omega_body_rad_s@I@x.omega_body_rad_s/2, atol=1e-12)
    velocity_ned = jnp.einsum("tij,tj->ti",R,trace.velocity_body_m_s)
    v0 = quat.rotation_matrix(x)@x.velocity_body_m_s
    np.testing.assert_allclose(velocity_ned, np.broadcast_to(v0,velocity_ned.shape), atol=1e-10)
    np.testing.assert_allclose(y.position_ned_m, v0*10, atol=1e-9)
    np.testing.assert_allclose(quat.rotation_matrix(y), euler.rotation_matrix(z), atol=1e-10)
    np.testing.assert_allclose(jnp.linalg.norm(trace.attitude,axis=1),1,atol=3e-16)


def test_quaternion_passes_pitch_lock_euler_rejects():
    quat = RigidBody6DoF()
    p = mass_properties(1.,np.eye(3),[0,0,0])
    x = quat.initialize(omega_body_rad_s=[0,1,0])
    y,_ = simulate(quat,x,p,BodyLoads(jnp.zeros(3),jnp.zeros(3)),steps=200)
    np.testing.assert_allclose(quat.rotation_matrix(y),Rotation.from_euler("y",2.).as_matrix(),atol=2e-11)
    with pytest.raises(ValueError):
        RigidBody6DoF("euler321").initialize(euler_rad=[0,np.pi/2,0])
    e = RigidBody6DoF("euler321")
    bad = e.initialize()._replace(attitude=jnp.array([0.,np.pi/2,0.]))
    assert np.any(np.isnan(e.rhs(bad,BodyLoads(jnp.zeros(3),jnp.zeros(3)),p).attitude))


def test_load_composition_and_stage_evaluation():
    load = sum_loads(force_at_point([0.,2,0],[1.,0,0]),force_at_point([0.,0,3],[0.,0,0]))
    np.testing.assert_array_equal(load.moment_body_Nm,[0,0,2])
    model = RigidBody6DoF()
    p = mass_properties(1.,np.eye(3),[0,0,0])
    x = model.initialize(velocity_body_m_s=[1,0,0])
    def drag(t,s,u,p,resources):
        return BodyLoads(-resources*s.velocity_body_m_s,jnp.zeros(3))
    # One RK4 interval for v'=-v: differs from incorrectly frozen initial loads.
    y = model.step(x,(),p,.5,load_fn=drag,resources=jnp.array(1.))
    assert float(y.velocity_body_m_s[0]) == pytest.approx(1-.5+.5**2/2-.5**3/6+.5**4/24)


def test_world_batch_rollout_and_gradient():
    from aerodrome.composition.rigid_body import RigidBodyAssembly
    from aerodrome.composition import EntitySpec,WorldSpec,build_world
    from aerodrome.runners.batch import BatchedWorld
    from aerodrome.core.clock import Schedule
    model = RigidBody6DoF()
    world = build_world(WorldSpec((EntitySpec("body",RigidBodyAssembly(model)),),
                                 Schedule(physics_dt_s=.01),ticks_per_step=2))
    initial = model.initialize()
    params = world.parameters({"body":mass_properties(2.,np.eye(3),[0,0,0])})
    loads = (BodyLoads(jnp.array([2.,0,0]),jnp.zeros(3)),)
    s = world.reset(0,{"body":initial})
    world.validate(s,loads,params)
    batch = BatchedWorld(world,initial_axes=None,input_axes=None)
    # Use the public batch reset API, including stable world IDs.
    state = batch.reset(0,jnp.array([0,1],jnp.uint32),{"body":initial})
    result,_ = jax.jit(lambda x:batch.rollout_constant(x,loads,params,steps=50,record=None))(state)
    np.testing.assert_allclose(result.world.entities[0].position_ned_m[:,0],.5,atol=1e-13)
    def position(force):
        y,_ = world.rollout(s,(BodyLoads(jnp.array([force,0,0]),jnp.zeros(3)),),params,steps=50)
        return y.entities[0].position_ned_m[0]
    assert float(jax.jit(jax.grad(position))(2.)) == pytest.approx(.25,abs=1e-12)


@pytest.mark.parametrize("mode",["quaternion","euler321"])
@pytest.mark.parametrize("dtype",[jnp.float32,jnp.float64])
def test_applied_torque_and_time_dependent_force(mode,dtype):
    model = RigidBody6DoF(mode)
    p = jax.tree.map(lambda x:x.astype(dtype),mass_properties(2.,np.diag([2.,3.,4.]),[0,0,0]))
    x = jax.tree.map(lambda x:x.astype(dtype),model.initialize())
    def law(t,s,u,p,resources):
        # F_x=2t keeps translation along the roll axis, M_x=2 -> p_dot=1.
        return BodyLoads(jnp.stack((2*t,jnp.zeros_like(t),jnp.zeros_like(t))),jnp.array([2.,0,0],dtype))
    def run(x):
        def tick(s,k):
            y = model.step(s,(),p,jnp.asarray(.01,dtype),time_s=k.astype(dtype)*.01,load_fn=law)
            return y,()
        return jax.lax.scan(tick,x,jnp.arange(100))[0]
    y = jax.jit(run)(x)
    assert all(v.dtype == dtype for v in y)
    tol = 2e-6 if dtype == jnp.float32 else 1e-10
    np.testing.assert_allclose(y.position_ned_m,[1/6,0,0],atol=tol)
    np.testing.assert_allclose(y.velocity_body_m_s,[.5,0,0],atol=tol)
    np.testing.assert_allclose(y.omega_body_rad_s,[1,0,0],atol=tol)
    np.testing.assert_allclose(model.rotation_matrix(y),Rotation.from_euler("x",.5).as_matrix(),atol=tol)


def test_f16_planar_reduction_matches_generic_body():
    from aerodrome.models.f16_longitudinal import Airframe,load_tables,coefficients,density,rhs
    p = Airframe()
    tables = load_tables()
    model = RigidBody6DoF("euler321")
    params = mass_properties(p.mass,np.diag([p.iyy,p.iyy,p.iyy]),[0,0,p.gravity])
    for alpha,theta,q in [(0.06,.08,.02),(-.02,.1,-.03)]:
        speed,h,elevator,thrust = 155.,3000.,-.02,9000.
        cx,cz,cm = coefficients(speed,alpha,q,elevator,tables,p)
        scale = .5*density(h)*speed**2*p.area
        load = BodyLoads(jnp.array([scale*cx+thrust,0,scale*cz]),jnp.array([0,scale*p.chord*cm,0]))
        x = model.initialize(position_ned_m=[0,0,-h],velocity_body_m_s=[speed*np.cos(alpha),0,speed*np.sin(alpha)],
                             euler_rad=[0,theta,0],omega_body_rad_s=[0,q,0])
        dx = model.rhs(x,load,params)
        u,_,w = x.velocity_body_m_s
        du,_,dw = dx.velocity_body_m_s
        reduced = jnp.array([(u*du+w*dw)/speed,(u*dw-w*du)/speed**2,
                             dx.omega_body_rad_s[1],dx.attitude[1],-dx.position_ned_m[2]])
        np.testing.assert_allclose(reduced,rhs(jnp.array([speed,alpha,q,theta,h]),jnp.array([elevator,thrust]),tables,p),atol=1e-12)
