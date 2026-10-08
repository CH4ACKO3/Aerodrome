"""One physical update, checked against analytic motion and JAX sensitivities."""
import jax
import jax.numpy as jnp
import numpy as np

from aerodrome.models.point_mass import PointMassState, step


def test_numpy_batch_matches_constant_acceleration_motion():
    initial = PointMassState(np.array([[1., 2.], [-3., 4.]]), np.array([[2., -1.], [.5, 3.]]))
    acceleration = np.array([[.3, -.2], [-1., .7]])
    state = initial
    for _ in range(50):
        state = step(state, acceleration, .04)
    np.testing.assert_allclose(state.position_m, initial.position_m+2*initial.velocity_m_s+2*acceleration, atol=1e-12)
    np.testing.assert_allclose(state.velocity_m_s, initial.velocity_m_s+2*acceleration, atol=1e-12)


def test_jax_scan_and_gradient_match_analytic_motion():
    initial = PointMassState(jnp.zeros(2), jnp.array([1., -1.]))
    def travel(acceleration):
        def tick(state, _):
            following = step(state, acceleration, .1)
            return following, following.position_m
        return jax.lax.scan(tick, initial, None, length=20)[0].position_m
    acceleration = jnp.array([.4, -.2])
    np.testing.assert_allclose(jax.jit(travel)(acceleration), 2*initial.velocity_m_s+2*acceleration, atol=1e-12)
    np.testing.assert_allclose(jax.jacfwd(travel)(acceleration), 2*np.eye(2), atol=1e-12)
