from aerodrome.simulator.CanonicalAircraftEnv.objects.WingedCone2D_Classic import WingedCone2D_Classic
from aerodrome.simulator.Core.envs.Space3D import Space3D
from math import *

if __name__ == "__main__":
    dt = 0.01
    env = Space3D(dt, 0.001)
    object_dict = {
        "name": "test",
        "integrator": "euler",
        "dt": 0.001,

        "S": 3603.0,
        "c": 80.0,
        "m": 9375.0,

        "pos": [0.0, 33528.0, 0.0],
        "vel": [4590.29, 0.0, 0.0],
        "ang_vel": [0.0, 0.0, 0.0],
        "J": [1.0*10**6, 0, 0, 0, 7*10**6, 0, 0, 0, 7*10**6],
        "theta": 0.00/180*pi,
        "phi": 0.0,
        "gamma": 0.0,   
        "theta_v": 0.0,
        "phi_v": 0.0,

        "Kiz": 0.2597,
        "Kwz": 1.6,
        "Kaz": 13/2,
        "Kpz": 0.14,
        "Kp_V": 5.0,
        "Ki_V": 1.0,
        "Kd_V": 0.3
    }

    object = WingedCone2D_Classic(object_dict)
    env.add_object(object)
    # result = env.to_dict()
    # print(result)

    import numpy as np

    cnt = 20000
    x = np.arange(cnt) * dt
    y = np.zeros(cnt)

    Nyc = 0.0
    Vc = 3000

    for i in range(cnt):
        action = {"test": {"Nyc":Nyc, "Vc":Vc}}
        result = env.step(action)
        y[i] = result["test"]["Ny"]

    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.xlabel("Time (s)")
    plt.ylabel("Ny")
    plt.axhline(y=Nyc, color='r', linestyle='--', alpha=0.5)
    plt.show()

    print(env.to_dict())