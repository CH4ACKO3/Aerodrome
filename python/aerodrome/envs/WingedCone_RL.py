from aerodrome.core import Env
from aerodrome.registration import register
from aerodrome.simulator.Core.envs.Space3D import Space3D
from copy import deepcopy
import numpy as np

class WingedCone_RL(Env):
    def __init__(self, dt=0.005):
        self.env = Space3D(dt, 0.001)
        self.object_name = None
        self.steps = 0
        self.eNy_bound = 1.0
        self.d_eNy_ = 0.0

    def add_object(self, object):
        self.env.add_object(object)
        self.object_name = object.to_dict()["name"]

    def reset(self):
        self.env.reset()
        state = self.env.to_dict()[self.object_name]
        obs = np.array([0.0, 0.0, 0.0])
        self.d_eNy_ = state["d_eNy"]

        return obs, {}

    def step(self, action):
        state = self.env.step(action)[self.object_name]
        obs = np.array([action[self.object_name]["Nyc"], state["eNy"], (state["d_eNy"] if self.steps>1 else 0.0)])
        self.alpha_ = state["alpha"]

        e = np.abs(state["eNy"]) / self.eNy_bound

        reward = -np.tanh(e)+1

        # if state["eNy"]*state["d_eNy"]<0:
        #     reward += 0.1
        # else:
        #     reward -= np.tanh(e)
        # reward += np.clip((-state["d_eNy"]/state["eNy"])-0.5, -1.0, 1.0)

        # if self.d_eNy_*state["d_eNy"]<0:
        #     reward -= 1.0
        self.d_eNy_ = state["d_eNy"]

        self.steps += 1

        if state["alpha"] > 88*np.pi/180 or state["alpha"] < -88*np.pi/180 or np.abs(state["eNy"]) > 20.0:
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
    
register("wingedcone-v0", "aerodrome.envs.WingedCone_RL:WingedCone_RL")