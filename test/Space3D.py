from aerodrome.core import Env
from aerodrome.registration import register
from aerodrome.envs.c_envs.Space3D import Space3D, Object3D
from math import *

env = Space3D(0.01, 0.001, "rk45", 1)
object_dict = {
    "name": "test",
    "pos": [0.0, 0.0, 0.0],
    "vel": [1.0, 1.0, 0.0],
    "acc": [100.0, 0.0, 0.0],
    "ang_vel": [0.0, 0.0, 0.0],
    "ang_acc": [0.0, 0.0, 0.0],
    "theta": 0.0,
    "phi": 0.0,
    "gamma": 0.0,   
    "theta_v": 45.0/180.0*pi,
    "phi_v": 0.0,
    "gamma_v": 0.0,
    "alpha": -45.0/180.0*pi,
    "beta": 0.0
}

object = Object3D(object_dict)
print(object.to_dict())

env.add_object(object)
print(env.to_dict())

for i in range(1000):
    env.kinematics_step()
print(env.to_dict())
