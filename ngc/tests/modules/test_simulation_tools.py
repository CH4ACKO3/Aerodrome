import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from aerodrome.core import rotations as rot, frames, airdata as air, units


@pytest.mark.parametrize("dtype",[jnp.float32,jnp.float64])
def test_quaternion_composition_inverse_and_rotvec(dtype):
    rng = np.random.default_rng(72)
    vectors = jnp.asarray(rng.normal(size=(20,3)),dtype)
    q = jax.jit(jax.vmap(rot.rotation_vector_to_quaternion))(vectors)
    matrices = jax.vmap(rot.quaternion_to_matrix)(q)
    tol = 5e-7 if dtype == jnp.float32 else 1e-13
    np.testing.assert_allclose(matrices,Rotation.from_rotvec(np.asarray(vectors)).as_matrix(),atol=tol)
    a,b = q[:10],q[10:]
    product = jax.vmap(rot.quaternion_multiply)(a,b)
    np.testing.assert_allclose(jax.vmap(rot.quaternion_to_matrix)(product),matrices[:10]@matrices[10:],atol=tol)
    identity = jax.vmap(lambda x:rot.quaternion_multiply(x,rot.quaternion_inverse(x)))(3*q)
    np.testing.assert_allclose(identity,np.broadcast_to([1,0,0,0],identity.shape),atol=tol)
    error = jax.vmap(rot.quaternion_error)(a,b)
    np.testing.assert_allclose(jax.vmap(rot.quaternion_to_matrix)(error),matrices[:10]@matrices[10:].transpose(0,2,1),atol=tol)
    assert q.dtype == dtype


def test_rotvec_zero_jacobian_and_skew():
    np.testing.assert_allclose(rot.rotation_vector_to_quaternion(jnp.zeros(3)),[1,0,0,0])
    jac = jax.jit(jax.jacfwd(rot.rotation_vector_to_quaternion))(jnp.zeros(3))
    np.testing.assert_allclose(jac,np.vstack((np.zeros(3),.5*np.eye(3))),atol=1e-14)
    a,b = jnp.array([1.,2,3]),jnp.array([-2.,4,1])
    np.testing.assert_allclose(rot.skew(a)@b,np.cross(a,b))


def test_euler_rates_match_rotation_derivative():
    e = jnp.array([.3,-.4,.7])
    omega = jnp.array([.2,-.5,.1])
    rates = rot.body_rates_to_euler321_rates(e,omega)
    dR = jax.jvp(rot.euler321_to_matrix,(e,),(rates,))[1]
    np.testing.assert_allclose(dR,rot.euler321_to_matrix(e)@rot.skew(omega),atol=1e-14)
    np.testing.assert_allclose(rot.euler321_rates_to_body_rates(e,rates),omega,atol=1e-14)


def test_point_vector_composition_and_wrench():
    R = rot.euler321_to_matrix(jnp.array([0.,0.,jnp.pi/2]))
    transform = frames.Transform(R,jnp.array([3.,0,0]))
    point = jnp.array([1.,0,0])
    np.testing.assert_allclose(frames.transform_vector(R,point),[0,1,0],atol=1e-14)
    np.testing.assert_allclose(frames.transform_point(transform,point),[3,1,0],atol=1e-14)
    inverse = frames.inverse_transform(transform)
    np.testing.assert_allclose(frames.transform_point(inverse,frames.transform_point(transform,point)),point,atol=1e-14)
    other = frames.Transform(rot.euler321_to_matrix(jnp.array([.2,-.3,.1])),jnp.array([2.,-3,1]))
    np.testing.assert_allclose(frames.transform_point(frames.compose_transforms(transform,other),point),
                               frames.transform_point(transform,frames.transform_point(other,point)),atol=1e-14)
    force,moment = jax.jit(frames.transform_wrench)(transform,jnp.array([2.,0,0]),jnp.array([0.,0,1]))
    np.testing.assert_allclose(force,[0,2,0],atol=1e-14)
    np.testing.assert_allclose(moment,[0,0,7],atol=1e-14)
    f,m = frames.transform_wrench(inverse,force,moment)
    np.testing.assert_allclose(f,[2,0,0],atol=1e-14)
    np.testing.assert_allclose(m,[0,0,1],atol=1e-14)
    cov = jnp.diag(jnp.array([1.,4,9]))
    np.testing.assert_allclose(frames.transform_covariance(R,cov),np.diag([4,1,9]),atol=1e-14)


def test_sensor_lever_arm_kinematics():
    v = frames.point_velocity(jnp.zeros(3),jnp.array([0.,0,2]),jnp.array([3.,0,0]))
    a = frames.point_acceleration(jnp.zeros(3),jnp.array([0.,0,2]),jnp.array([0.,0,1]),jnp.array([3.,0,0]))
    np.testing.assert_allclose(v,[0,6,0])
    np.testing.assert_allclose(a,[-12,3,0])


def test_local_geographic_frames_and_axis_flips():
    # Equator, Greenwich: North=ECEF z, East=ECEF y, Down=-ECEF x.
    R = frames.ecef_to_ned_matrix(0.,0.)
    np.testing.assert_allclose(R,[[0,0,1],[0,1,0],[-1,0,0]])
    origin = jnp.array([6378137.,0,0])
    transform = frames.ecef_to_ned_transform(origin,0.,0.)
    np.testing.assert_allclose(frames.transform_point(transform,origin+jnp.array([3.,4,5])),[5,4,-3])
    angles = jnp.array([[.4,-.7],[-.2,2.],[jnp.pi/2,0.]])
    Rs = jax.jit(jax.vmap(lambda x:frames.ecef_to_ned_matrix(*x)))(angles)
    np.testing.assert_allclose(Rs@Rs.transpose(0,2,1),np.broadcast_to(np.eye(3),Rs.shape),atol=1e-14)
    np.testing.assert_allclose(jnp.linalg.det(Rs),1,atol=1e-14)
    v = jnp.array([1.,2,3])
    np.testing.assert_array_equal(frames.ned_to_enu(v),[2,1,-3])
    np.testing.assert_array_equal(frames.enu_to_ned(frames.ned_to_enu(v)),v)
    np.testing.assert_array_equal(frames.frd_to_flu(v),[1,-2,-3])
    np.testing.assert_array_equal(frames.flu_to_frd(frames.frd_to_flu(v)),v)


def test_airdata_wind_axes_and_force_signs():
    queries = jnp.array([[100.,.15,.08],[40.,-.3,-.2],[150.,0.,0.]])
    velocities = jax.jit(jax.vmap(lambda x:air.velocity_from_airdata(*x)))(queries)
    data = jax.jit(jax.vmap(air.airdata))(velocities)
    np.testing.assert_allclose(jnp.stack(data[:3],axis=1),queries,atol=1e-13)
    assert np.all(data.angles_valid)
    R = jax.vmap(lambda x:air.body_to_wind_matrix(x[1],x[2]))(queries)
    np.testing.assert_allclose(jnp.einsum("bij,bj->bi",R,velocities),np.column_stack((queries[:,0],np.zeros((3,2)))),atol=1e-13)
    np.testing.assert_allclose(R@R.transpose(0,2,1),np.broadcast_to(np.eye(3),R.shape),atol=1e-14)
    np.testing.assert_allclose(air.aerodynamic_force_body(10.,2.,100.,0.,0.),[-10,2,-100])
    f = air.aerodynamic_force_body(10.,0.,100.,jnp.pi/2,0.)
    np.testing.assert_allclose(f,[100,0,-10],atol=1e-13)
    assert float(air.dynamic_pressure(1.2,100.)) == pytest.approx(6000.)


def test_wind_subtraction_course_and_invalid_air_angles():
    # Body x points East; eastward tailwind subtracts 10 m/s from body x.
    R = rot.euler321_to_matrix(jnp.array([0.,0.,jnp.pi/2]))
    np.testing.assert_allclose(air.air_relative_velocity(jnp.array([100.,0,0]),jnp.array([0.,10,0]),R),[90,0,0],atol=1e-13)
    course,climb = air.flight_path_angles(jnp.array([0.,10,-10]))
    np.testing.assert_allclose([course,climb],[np.pi/2,np.pi/4])
    for v in [jnp.zeros(3),jnp.array([0.,20,0])]:
        result = air.airdata(v)
        assert not bool(result.angles_valid)
        assert np.isnan(result.alpha_rad) and np.isnan(result.beta_rad)
    assert np.all(np.isnan(air.flight_path_angles(jnp.zeros(3))))
    vertical = air.flight_path_angles(jnp.array([0.,0.,-10.]))
    assert np.isnan(vertical[0])
    assert float(vertical[1]) == pytest.approx(np.pi/2)


def test_units_wrapping_and_airdata_gradient():
    assert float(units.feet_to_m(1.)) == pytest.approx(.3048)
    assert float(units.knots_to_m_s(3600.)) == pytest.approx(1852.)
    np.testing.assert_allclose(units.m_s_to_knots(units.knots_to_m_s(jnp.array([1.,20]))),[1,20])
    np.testing.assert_allclose(units.m_to_feet(units.feet_to_m(jnp.array([1.,20]))),[1,20])
    assert float(units.angle_difference(jnp.deg2rad(-179.),jnp.deg2rad(179.))) == pytest.approx(np.deg2rad(2))
    assert float(units.wrap_pi(jnp.pi)) == pytest.approx(-np.pi)
    assert float(units.wrap_2pi(-.2)) == pytest.approx(2*np.pi-.2)
    x = jnp.array([100.,4.,12.])
    fn = lambda v:jnp.stack(air.airdata(v)[:3])
    jac = jax.jit(jax.jacfwd(fn))(x)
    h = 1e-3
    finite = np.stack([(fn(x+jnp.eye(3)[i]*h)-fn(x-jnp.eye(3)[i]*h))/(2*h) for i in range(3)],axis=1)
    np.testing.assert_allclose(jac,finite,atol=1e-10,rtol=1e-8)
