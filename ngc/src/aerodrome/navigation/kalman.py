"""Linear Kalman filter with Joseph covariance update.

Host setup computes exact ZOH matrices and discretizes continuous model
uncertainty; predict/correct themselves are pure JAX functions.
"""
from typing import NamedTuple
import numpy as np
from scipy.linalg import expm
import jax
import jax.numpy as jnp
from aerodrome.core.signals import Array, GaussianState, to_vector, from_vector


class KalmanParameters(NamedTuple):
    transition: Array
    input_matrix: Array
    process_covariance: Array
    measurement_variance: Array


def discretize(nominal_model, dt_s, acceleration_noise_density, sensor_std_rad):
    """kf.discretize: Qd = integral exp(A*t) W exp(A.T*t) dt.

    Density is angular-acceleration white-noise spectral intensity [rad^2/s^3].
    It is an estimator model-error assumption; the demo plant is deterministic.
    """
    a, b, c = map(float, nominal_model)
    values = (a, b, c, dt_s, acceleration_noise_density, sensor_std_rad)
    if not all(np.isfinite(values)) or dt_s <= 0 or acceleration_noise_density < 0 or sensor_std_rad <= 0:
        raise ValueError("Invalid discretization, covariance or sensor parameters")
    A = np.array([[0., 1.], [-a, -b]])
    B = np.array([[0.], [c]])
    augmented = np.zeros((3, 3))
    augmented[:2, :2], augmented[:2, 2:] = A, B
    discrete = expm(augmented * dt_s)
    F, G = discrete[:2, :2], discrete[:2, 2:]
    W = np.diag([0., acceleration_noise_density])
    van_loan = np.block([[A, W], [np.zeros((2, 2)), -A.T]])
    exponential = expm(van_loan * dt_s)
    Q = exponential[:2, 2:] @ F.T
    return KalmanParameters(*map(jnp.asarray, (F, G, (Q+Q.T)/2, sensor_std_rad**2)))


def predict(state, known_actuator, params):
    """kf.predict: x-=F*x+G*u; P-=F*P*F.T+Qd."""
    F, G, Q, _ = params
    mean = F @ to_vector(state.mean) + G[:, 0] * known_actuator.elevator_rad
    covariance = F @ state.covariance @ F.T + Q
    return GaussianState(from_vector(mean), (covariance + covariance.T) / 2)


def correct(prior, measurement, params):
    """kf.correct: H=[1,0]. No update when the sample is invalid."""
    def update():
        P = prior.covariance
        H = jnp.array([1., 0.], dtype=P.dtype)
        innovation = measurement.pitch_rad - prior.mean.pitch_rad
        variance = H @ P @ H + params.measurement_variance
        gain = P @ H / variance
        mean = to_vector(prior.mean) + gain * innovation
        residual = jnp.eye(2, dtype=P.dtype) - jnp.outer(gain, H)
        covariance = residual @ P @ residual.T + jnp.outer(gain, gain) * params.measurement_variance
        return GaussianState(from_vector(mean), (covariance + covariance.T) / 2)
    return jax.lax.cond(measurement.valid, update, lambda: prior)
