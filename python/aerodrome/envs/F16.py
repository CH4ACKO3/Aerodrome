from aerodrome.core import Env
from aerodrome.registration import register
from aerodrome.simulator.Core.envs.Space3D import Space3D
from copy import deepcopy
import numpy as np

class WingedCone_RL(Env):
    def __init__(self, dt):
        self.env = Space3D(dt, 0.001)
        self.object_name = None

        self.eNy = 0
        self.i_eNy = 0
        self.d_eNy = 0
        self.eNz = 0
        self.i_eNz = 0
        self.d_eNz = 0
        self.eGamma = 0
        self.i_eGamma = 0
        self.d_eGamma = 0

    def add_object(self, object):
        self.env.add_object(object)
        self.object_name = object.to_dict()["name"]

    def reset(self):
        self.env.reset()
        state = self.env.to_dict()[self.object_name]
        obs = np.array([state["h"], state["V"], state["vel_b"][0], state["vel_b"][1], state["vel_b"][2],
                        state["ang_vel_b"][0], state["ang_vel_b"][1], state["ang_vel_b"][2],
                        state["alpha"], state["beta"], state["gamma"],
                        state["theta_v"], state["theta"], state["Ny"], state["Nz"]])
        ctrl_obs = np.array([state["u_deg"][0], state["u_deg"][1], state["u_deg"][2], state["u_deg"][3]])

        return obs, {}

    def step(self, action):
        state = self.env.step(action)[self.object_name]
        obs = np.array([state["eNy"], state["i_eNy"], state["d_eNy"]])
        # if state["eNy"] > self.eNy_bound:
        #     if state["d_eNy"] < 0:
        #         reward = -np.tanh(np.abs(state["eNy"]) / self.eNy_bound) + 1
        #     else:
        #         reward = -np.tanh(np.abs(state["eNy"]) / self.eNy_bound)
        # elif state["eNy"] < -self.eNy_bound:
        #     if state["d_eNy"] > 0:
        #         reward = -np.tanh(np.abs(state["eNy"]) / self.eNy_bound) + 1
        #     else:
        #         reward = -np.tanh(np.abs(state["eNy"]) / self.eNy_bound)
        # else:
        #     reward = -np.tanh(np.abs(state["eNy"]) / self.eNy_bound) + 1

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
    
register("wingedcone-v0", "aerodrome.envs.WingedCone_RL:WingedCone_RL")