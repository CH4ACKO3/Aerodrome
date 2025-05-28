from math import pi
from aerodrome.core import Env
from aerodrome.registration import register
from aerodrome.simulator.Core.envs.Space3D import Space3D
from copy import deepcopy
import numpy as np

class F16(Env):
    def __init__(self, dt=0.005):
        self.env = Space3D(dt, 0.001)
        self.object_name = None
        self.steps = 0

        self.eNz = 0
        self.ePs = 0
        self.eNy = 0

    def add_object(self, object):
        self.env.add_object(object)
        self.object_name = object.to_dict()["name"]

    def reset(self):
        self.env.reset()
        state = self.env.to_dict()[self.object_name]
        obs = np.array([state["h"]/1000.0, state["V"]/1000.0, state["vel_b"][0]/1000.0, state["vel_b"][1]/1000.0, state["vel_b"][2]/1000.0,
                        state["ang_vel_b"][0], state["ang_vel_b"][1], state["ang_vel_b"][2],
                        state["alpha"], state["beta"], state["gamma"],
                        state["theta_v"], state["theta"], state["Ny"], state["Nz"], self.eNz, self.ePs, self.eNy])

        return obs, {}

    def step(self, action):
        state = self.env.step(action)[self.object_name]
        self.eNz = action[self.object_name]["Nz"] - state["Nz"]
        self.ePs = action[self.object_name]["ps"] - state["gamma"]
        self.eNy = action[self.object_name]["Ny"] - state["Ny"]

        obs = np.array([state["h"]/1000.0, state["V"]/1000.0, state["vel_b"][0]/1000.0, state["vel_b"][1]/1000.0, state["vel_b"][2]/1000.0,
                        state["ang_vel_b"][0], state["ang_vel_b"][1], state["ang_vel_b"][2],
                        state["alpha"], state["beta"], state["gamma"],
                        state["theta_v"], state["theta"], state["Ny"], state["Nz"], self.eNz, self.ePs, self.eNy])

        reward1 = -np.tanh(np.abs(self.eNz) / 1.0) + 1
        reward2 = -np.tanh(np.abs(self.eNy) / 1.0) + 1
        reward3 = -np.tanh(np.abs(self.ePs) / (5.0/180.0*pi)) + 1

        reward = sum([reward1, reward2, reward3])
        self.steps += 1
        terminate_list = [
            state["alpha"] > 45.0/180.0*pi,
            state["alpha"] < -10.0/180.0*pi,
            state["beta"] > 30.0/180.0*pi,
            state["beta"] < -30.0/180.0*pi,
        ]
        if any(terminate_list):
            terminated = np.array([1], dtype=np.bool_)
            reward -= 10.0
        else:
            terminated = np.array([0], dtype=np.bool_)

        if self.steps > 1024:
            truncated = np.array([1], dtype=np.bool_)
        else:
            truncated = np.array([0], dtype=np.bool_)    
        
        return obs, reward*0.1, terminated, truncated, {}
    
    def get_state(self):
        state = self.env.to_dict()[self.object_name]
        return state
    
register("F16-v0", "aerodrome.envs.F16:F16")