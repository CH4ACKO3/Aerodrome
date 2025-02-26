from aerodrome.core import Env
from aerodrome.registration import register
from aerodrome.simulator.objects.WingedCone2D_control import WingedCone2D_control
from aerodrome.simulator.envs.Space3D import Space3D
from math import *

for dt in [0.05]:

    env = Space3D(dt, 0.001, 1)
    object_dict = {
        "name": "test",
        "integrator": "euler",
        "S": 3603.0,
        "c": 80.0,
        "m": 9375.0,

        "pos": [0.0, 33528.0, 0.0],
        "vel": [4590.29, 0.0, 0.0],
        "ang_vel": [0.0, 0.0, 0.0],
        "J": [1.0, 7*10**6, 7*10**6],
        "theta": 0.00/180*pi,
        "phi": 0.0,
        "gamma": 0.0,   
        "theta_v": 0.0,
        "phi_v": 0.0,
        "gamma_v": 0.0,
        "alpha": 0.00/180*pi,
        "beta": 0.0,

        "Kiz": 0.2597,
        "Kwz": 1.6,
        "Kaz": 13/2,
        "Kpz": 0.14,
        "Kp_V": 5.0,
        "Ki_V": 1.0,
        "Kd_V": 0.3
    }

    object = WingedCone2D_control(object_dict)
    print(object.to_dict())

    env.add_object(object)
    # print(env.to_dict())

    print("----------------------------------")
    print(env.get_d())
    print("----------------------------------")

    import numpy as np
    import matplotlib.pyplot as plt

    cnt = 1000
    x = np.zeros(cnt)
    y = np.zeros(cnt)

    for i in range(cnt):
        action = {"test": {"Nyc":0.0, "Vc":3000}}
        result = env.step(action)
        x[i] = result["test"]["pos"][0]
        y[i] = result["test"]["Ny"]

    plt.plot(x, y, label=f"dt={dt}")
    # plt.ylim(0, 40000)
    # plt.gca().set_aspect('equal', adjustable='box')

plt.legend()
plt.show()

print("----------------------------------")
print(env.get_d())
print("----------------------------------")
print(env.to_dict())
