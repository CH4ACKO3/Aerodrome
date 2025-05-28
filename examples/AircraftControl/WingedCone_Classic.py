from aerodrome.simulator.CanonicalAircraftEnv.objects.WingedCone2D_Classic import WingedCone2D_Classic
from aerodrome.simulator.Core.envs.Space3D import Space3D
from math import *

if __name__ == "__main__":
    dt = 0.005
    env = Space3D(dt, 0.001)
    object_dict = {
        "name": "test",
        "integrator": "rk4",
        "dt": dt,

        "S": 3603.0,
        "c": 80.0,
        "m": 9375.0,

        "pos": [0.0, 0.0, -33528.0],
        "vel": [4590.29, 0.0, 0.0],
        "ang_vel": [0.0, 0.0, 0.0],
        "J": [1.0*10**6, 0, 0, 0, 7*10**6, 0, 0, 0, 7*10**6],
        "theta": -0.2/180*pi,
        "phi": 0.0,
        "gamma": 0.0,   
        "theta_v": 0.0,
        "phi_v": 0.0,

        "Kiz": 0.2597 * 0.1,
        "Kwz": 1.6 ,
        "Kaz": 13/2 * 0.5,
        "Kpz": 0.14 * 0.1,
        "Kp_V": 5.0,
        "Ki_V": 1.0,
        "Kd_V": 0.3
    }

    object = WingedCone2D_Classic(object_dict)
    env.add_object(object)
    # result = env.to_dict()
    # print(result)

    import numpy as np

    cnt = 1
    x = np.arange(cnt) * dt
    y = np.zeros(cnt)

    Nyc = 1.0
    Vc = 3000

    for i in range(cnt):
        action = {"test": {"Nyc":Nyc, "Vc":Vc}}
        result = env.step(action)
        # y[i] = sum(result["test"]["quat"] ** 2)
        y[i] = result["test"]["Ny"]

    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(x, y, c="tab:blue")
    plt.xlabel("时间 $s$")
    plt.ylabel("过载 $N_y$")
    plt.title("纵向过载阶跃响应")
    plt.axhline(y=Nyc, color='r', linestyle='--', alpha=0.5)
    plt.show()

    # np.save("wingedcone_classic.npy", y)
    print(env.to_dict())