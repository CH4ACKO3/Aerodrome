from aerodrome.core import Env
from aerodrome.registration import register
from aerodrome.simulator.objects.WingedCone2D_control import WingedCone2D_control
from aerodrome.simulator.objects.Object3D import Object3D, add
from aerodrome.simulator.envs.Space3D import Space3D
from math import *


env = Space3D(0.001, 0.001, 1)
object_dict = {
    "name": "test",
    "integrator": "rk45",
    "S": 3603.0,
    "c": 80.0,
    "m": 9375.0,

    "pos": [0.0, 33528.0, 0.0],
    "vel": [4590.29, 0.0, 0.0],
    "ang_vel": [0.0, 0.0, 0.0],
    "J": [1.0, 7*10**6, 7*10**6],
    "theta": 3/180*pi,
    "phi": 0.0,
    "gamma": 0.0,   
    "theta_v": 0.0,
    "phi_v": 0.0,
    "gamma_v": 0.0,
    "alpha": 3/180*pi,
    "beta": 0.0,

    "Kiz": 0.2597,
    "Kwz": 1.6,
    "Kaz": 13/2,
    "Kpz": 0.14,
    "Kp_V": 5.0,
    "Ki_V": 1.0,
    "Kd_V": 0.3
}

object1 = Object3D(object_dict)
object2 = Object3D(object_dict)

print(object1.to_dict())
print(type(object1))

print(add(object1, object2).to_dict())
print(type(add(object1, object2)))