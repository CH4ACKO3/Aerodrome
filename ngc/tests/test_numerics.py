import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import expm
from aerodrome.models.pitch import PitchParameters, advance
from aerodrome.core.signals import PitchState, ElevatorPosition, to_vector
from aerodrome.navigation.kalman import discretize


def test_rk4_against_analytic_solution_and_convergence():
    model = PitchParameters(2., 0.7, 3.)
    initial = PitchState(jnp.asarray(0.3), jnp.asarray(-0.2))
    actuator = ElevatorPosition(jnp.asarray(0.1))
    augmented = np.array([[0., 1., 0.], [-2., -0.7, 0.3], [0., 0., 0.]])
    expected = (expm(augmented) @ np.array([0.3, -0.2, 1.]))[:2]
    errors = []
    for steps in (10, 20):
        final, _ = jax.lax.scan(
            lambda s, _: (advance(s, actuator, model, 1 / steps), None),
            initial, None, length=steps)
        errors.append(np.linalg.norm(np.asarray(to_vector(final)) - expected))
    assert 12 < errors[0] / errors[1] < 20
    assert errors[1] < 1e-6


def test_noise_discretization_matches_integrated_white_acceleration():
    dt, density = 0.1, 0.02
    p = discretize(PitchParameters(0., 0., 1.), dt, density, 0.01)
    expected = density * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    np.testing.assert_allclose(p.process_covariance, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(p.transition, [[1., dt], [0., 1.]])
    np.testing.assert_allclose(p.input_matrix, [[dt**2 / 2], [dt]])
