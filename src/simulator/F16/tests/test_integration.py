"""Numerical regression for the compiled F16 and Object3D Python modules."""

import math
import unittest

import Object3D  # noqa: F401 - registers the pybind base class
from F16 import F16


def aircraft(model, dt, integrator):
    speed, alpha, beta = 600.0, 0.1, 0.04
    u = speed * math.cos(alpha) * math.cos(beta)
    v = speed * math.sin(beta)
    w = speed * math.sin(alpha) * math.cos(beta)
    inertia = [9496.0, 0.0, -982.0,
               0.0, 55814.0, 0.0,
               -982.0, 0.0, 63100.0]
    return F16(dict(
        name="f16", integrator=integrator, dt=dt, m=636.94,
        pos=[500.0, 200.0, -8000.0], vel=[u, v, w],
        ang_vel=[0.04, -0.02, 0.03], J=inertia,
        theta=0.0, phi=0.0, gamma=0.0,
        theta_v=math.asin(-w / speed), phi_v=math.atan2(-v, u),
        S=300.0, c=11.32, power=30.0, model=model,
    ))


def trajectory(model, dt, integrator):
    plane = aircraft(model, dt, integrator)
    controls = dict(thtlc=0.5, el=-3.0, ail=5.0, rdr=-2.0)
    for _ in range(round(0.2 / dt)):
        result = plane.step(controls)
    return [*result["pos"], *result["vel_b"], *result["ang_vel_b"],
            *result["quat"], result["power"]]


def distance(first, second):
    return max(abs(a - b) for a, b in zip(first, second))


class F16IntegrationTest(unittest.TestCase):
    def test_integrators_converge_at_their_expected_orders(self):
        for model in ("stevens", "morelli"):
            with self.subTest(model=model):
                reference = trajectory(model, 0.0025, "rk4")
                for integrator, improvement in (("euler", 1.8),
                                                ("midpoint", 3.5),
                                                ("rk4", 10.0)):
                    coarse = distance(trajectory(model, 0.04, integrator), reference)
                    fine = distance(trajectory(model, 0.02, integrator), reference)
                    self.assertLess(fine, coarse / improvement)
                    if integrator == "rk4":
                        self.assertLess(fine, 1e-5)


if __name__ == "__main__":
    unittest.main()
