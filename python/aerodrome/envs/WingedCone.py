from aerodrome.core import Env
from aerodrome.registration import register
from aerodrome.simulator.Core.envs.Space3D import Space3D
from copy import deepcopy
import numpy as np

class WingedCone_RL(Env):
    def __init__(self):
        self.env = Space3D(0.01, 0.001, 5)
        self.env_copy = Space3D(0.01, 0.001, 5)
        self.object_name = None
        self.eNy_bound = 1

    def add_object(self, object):
        self.env.add_object(object)
        self.env_copy.add_object(object)
        self.object_name = object.to_dict()["name"]

    def reset(self):
        self.env = self.env_copy
        state = self.env.to_dict()[self.object_name]
        obs = np.array([state["eNy"], state["i_eNy"], state["d_eNy"]])

        return obs, {}

    def step(self, action):
        state = self.env.step(action)[self.object_name]
        obs = np.array([state["eNy"], state["i_eNy"], state["d_eNy"]])
        if state["eNy"] > self.eNy_bound:
            if state["d_eNy"] < 0:
                reward = -np.tanh(np.abs(state["eNy"]) / self.eNy_bound) + 1
            else:
                reward = -np.tanh(np.abs(state["eNy"]) / self.eNy_bound)
        elif state["eNy"] < -self.eNy_bound:
            if state["d_eNy"] > 0:
                reward = -np.tanh(np.abs(state["eNy"]) / self.eNy_bound) + 1
            else:
                reward = -np.tanh(np.abs(state["eNy"]) / self.eNy_bound)
        else:
            reward = -np.tanh(np.abs(state["eNy"]) / self.eNy_bound) + 1

        if state["alpha"] > 88*np.pi/180 or state["alpha"] < -88*np.pi/180:
            terminated = np.array([1], dtype=np.bool_)
            reward -= 10.0
        else:
            terminated = np.array([0], dtype=np.bool_)
        
        return obs, reward, terminated, False, {}
    
    def get_state(self):
        state = self.env.to_dict()[self.object_name]
        return state
    
register("wingedcone-v0", "aerodrome.envs.WingedCone:WingedCone_RL")