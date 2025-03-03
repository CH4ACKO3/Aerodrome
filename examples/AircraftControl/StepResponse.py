import random
import numpy as np
from matplotlib import pyplot as plt
from math import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import aerodrome
from aerodrome.simulator.CanonicalAircraftEnv.objects.WingedCone2D_RL import WingedCone2D_RL
from aerodrome.simulator.CanonicalAircraftEnv.objects.WingedCone2D_Classic import WingedCone2D_Classic
from aerodrome.simulator.Core.envs.Space3D import Space3D

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(3, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(3, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=0.01),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 1))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, evaluate=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        if evaluate:
            probs = Normal(action_mean, action_std*1e-6)
        else:
            probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        action = torch.clamp(action, -1.0, 1.0)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    

def linear_schedule(start_e, end_e, duration, t):
    return start_e + (end_e - start_e) * min(t / duration, 1)

def main():
    fig, ax = plt.subplots(1, 1)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ny")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    
    env = aerodrome.make("wingedcone-v0")
    object = WingedCone2D_RL(object_dict)
    env.add_object(object)

    agent = Agent().to(device)
    agent.load_state_dict(torch.load("wingedcone_ppo.pth"))
    agent.eval()

    states = []
    rewards = []
    next_obs, info = env.reset()
    next_obs = torch.Tensor(next_obs).reshape((1, -1)).to(device)
    step = 0
    while True:
        step += 1
        states.append(env.get_state())
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs, evaluate=True)

        step_action = {
            "test": {"Nyc":1.0, "Vc":4590.29, "nn_control":action.item()},
        }
        next_obs, reward, terminations, truncations, infos = env.step(step_action)
        next_done = np.logical_or(terminations, truncations)
        next_obs = torch.Tensor(next_obs).reshape((1, -1)).to(device)
        rewards.append(reward)
        if next_done or step >= 1000:
            break
    
    x = np.arange(1000) * 0.01
    y = np.array([states[i]["Ny"] for i in range(len(states))])
    ax.plot(x, y, label="RL")

    # Classical control
    env = Space3D(0.01, 0.001, 5)
    object = WingedCone2D_Classic(object_dict)
    env.add_object(object)

    cnt = 1000
    y = np.zeros(cnt)

    for i in range(cnt):
        action = {"test": {"Nyc":1.0, "Vc":4590.29}}
        result = env.step(action)
        y[i] = result["test"]["Ny"]

    ax.plot(x, y, label="Classic")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()