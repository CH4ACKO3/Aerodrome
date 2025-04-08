import aerodrome
from aerodrome.simulator.CanonicalAircraftEnv.objects.WingedCone2D_Classic import WingedCone2D_Classic
from aerodrome.simulator.Core.envs.Space3D import Space3D

from math import *
import numpy as np
from time import time

import matplotlib.pyplot as plt

if __name__ == "__main__":
    dt = 0.001
    input_dict = {
        "name": "test",
        "integrator": "rk4",
        "dt": dt,
        "S": 3603.0,
        "c": 80.0,
        "m": 9375.0,

        "pos": [0.0, 33528.0, 0.0],
        "vel": [4590.29, 0.0, 0.0],
        "ang_vel": [0.0, 0.0, 0.0],
        "J": [1.0, 0, 0, 0, 7*10**6, 0,  0, 0, 7*10**6],
        "theta": 0.00/180*pi,
        "phi": 0.0,
        "gamma": 0.0,   
        "theta_v": 0.0,
        "phi_v": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "gamma_v": 0.0,

        "Kiz": 0.2597 * 0.01,
        "Kwz": 1.6,
        "Kaz": 13/2 * 0.1,
        "Kpz": 0.14 * 0.1,
        "Kp_V": 5.0,
        "Ki_V": 1.0,
        "Kd_V": 0.3
    }

    fig, ax = plt.subplots()

    # Python
    env = aerodrome.make("wingedcone_py-v0", input_dict)
    env.reset()

    cnt = 10000

    Nyc = 1.0
    Vc = 3000

    x = np.arange(cnt) * dt
    y = np.zeros(cnt)

    start_time = time()
    for i in range(cnt):
        action = {"Nyc":Nyc, "Vc":Vc}
        result_py = env.step(action)
        y[i] = result_py["Ny"]
    end_time = time()

    ax.plot(x, y, label=f"Python: {round(end_time - start_time, 2)}s")
    print(f"--------------Python: {round(end_time - start_time, 2)}s--------------")
    # print(env.to_dict())

    # C++
    env = Space3D(dt, 0.001)
    object = WingedCone2D_Classic(input_dict)
    env.add_object(object)

    start_time = time()
    for i in range(cnt):
        action = {"test": {"Nyc":Nyc, "Vc":Vc}}
        result_cpp = env.step(action)
        y[i] = result_cpp["test"]["Ny"]
    end_time = time()

    ax.plot(x, y, label=f"C++: {round(end_time - start_time, 2)}s")
    print(f"--------------C++: {round(end_time - start_time, 2)}s--------------")
    # print(env.to_dict()["test"])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ny")
    ax.set_title("Run Time Comparison")
    ax.legend()
    plt.show()

    print("Python CPP")
    for key in result_cpp["test"].keys():
        print(f"{key}: {result_py[key]}, {result_cpp['test'][key]}")