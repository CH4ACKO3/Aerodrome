"""Real World rollout of trimmed and excited airframes, batched and JIT compiled."""
import numpy as np

from f16_six_dof import simulate


def test_six_dof_surface_pulse_flight():
    records, summary = simulate()
    assert summary["max_trim_altitude_drift_m"] < 1e-8
    np.testing.assert_allclose(records["time_s"][[0, -1]], [0., 4.])
    np.testing.assert_allclose(np.linalg.norm(records["attitude"], axis=-1), 1., atol=1e-12)
    rates = np.rad2deg(records["omega_body_rad_s"])
    assert np.max(np.abs(rates[0])) < 1e-8
    assert np.all(np.max(np.abs(rates[1]), axis=0) > .5)
    np.testing.assert_allclose(records["euler_rad"][0, :51], records["euler_rad"][1, :51], atol=1e-12)
    assert np.linalg.norm(records["position_ned_m"][1, -1]-records["position_ned_m"][0, -1]) > .1
