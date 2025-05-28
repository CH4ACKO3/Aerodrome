import random
import numpy as np
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from math import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

import aerodrome
from aerodrome.simulator.CanonicalAircraftEnv.objects.WingedCone2D_RL import WingedCone2D_RL

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(4, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(4, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=0.001),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, 1) * np.log(0.001))

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed of the experiment")
    parser.add_argument("--num_steps", type=int, default=1024,
                        help="the number of steps to run per policy rollout")
    parser.add_argument("--total_timesteps", type=int, default=200_000,
                        help="the number of iterations")
    parser.add_argument("--gamma", type=float, default=0.995,
                        help="discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=1,
                        help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=1,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm_adv", type=bool, default=False,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=bool, default=False,
                        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent_coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="the target KL divergence threshold")
                        
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the experiment on")
    args = parser.parse_args()
    
    args.batch_size = args.num_steps
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = False

    device = torch.device(args.device)
    print(f"Using device: {device}")

    env = aerodrome.make("wingedcone-v0")
    dt = 0.005
    object_dict = {
        "name": "test",
        "integrator": "euler",
        "dt": dt,

        "S": 3603.0,
        "c": 80.0,
        "m": 9375.0,

        "pos": [0.0, 0.0, -33528.0],
        "vel": [4590.29, 0.0, 0.0],
        "ang_vel": [0.0, 0.0, 0.0],
        "J": [1.0*10**6, 0, 0, 0, 7*10**6, 0, 0, 0, 7*10**6],
        "theta": 0.00/180*pi,
        "phi": 0.0,
        "gamma": 0.0,   
        "theta_v": 0.0,
        "phi_v": 0.0,

        "Kiz": 0.2597 * 0.1,
        "Kwz": 1.6,
        "Kaz": 13/2*0.5,
        "Kpz": 0.14 * 0.1,
        "Kp_V": 5.0,
        "Ki_V": 1.0,
        "Kd_V": 0.3
    }

    object = WingedCone2D_RL(object_dict)
    env.add_object(object)

    agent = Agent().to(device)
    agent.load_state_dict(torch.load("models/wingedcone_ppo.pth"))
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    cos_annealing_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iterations, eta_min=args.learning_rate/10)

    # Storage setup
    obs = torch.zeros((args.num_steps, 4)).to(device)
    actions = torch.zeros((args.num_steps, 1)).to(device)
    logprobs = torch.zeros((args.num_steps, 1)).to(device)
    rewards = torch.zeros((args.num_steps, 1)).to(device)
    dones = torch.zeros((args.num_steps, 1)).to(device)
    values = torch.zeros((args.num_steps, 1)).to(device)

    global_step = 0

    records = {
        "reward": np.zeros(args.num_iterations),
        "learning_rate": np.zeros(args.num_iterations),
        "value_loss": np.zeros(args.num_iterations),
        "policy_loss": np.zeros(args.num_iterations),
        "entropy": np.zeros(args.num_iterations),
    }

    for iteration in tqdm(range(1, args.num_iterations + 1)):
        A = 0.00028
        object_dict["theta"] = object_dict["alpha"] = A
        # Nyc = np.random.uniform(-1.0, 1.0)
        Nyc = 0.0

        env = aerodrome.make("wingedcone-v0")
        object = WingedCone2D_RL(object_dict)
        env.add_object(object)

        next_obs, info = env.reset()
        next_obs = torch.Tensor(next_obs).reshape((1, -1)).to(device)
        next_done = torch.zeros((1, 1)).to(device)
        first_rollout = True
        rollout_return = 0.0
        c_step = 0
        for step in range(0, args.num_steps):
            obs[step] = next_obs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            if next_done:
                dones[step] = next_done
                next_done = False
                env = aerodrome.make("wingedcone-v0")

                A = np.random.uniform(-0.0001, 0.0003)
                Nyc = np.random.uniform(-0.5, 1.5) * iteration / args.num_iterations
                object_dict["theta"] = object_dict["alpha"] = A
                object = WingedCone2D_RL(object_dict)
                env.add_object(object)
                c_step = 0

                next_obs, info = env.reset()
                next_obs = torch.Tensor(next_obs).reshape((1, -1)).to(device)
                continue

            # TRY NOT TO MODIFY: execute the game and log data.
            step_action = {
                "test": {"Nyc":Nyc, "Vc":4590.29, "nn_control":action.item()},
            }
            next_obs, reward, terminations, truncations, infos = env.step(step_action)
            # next_done = np.logical_or(terminations, truncations)
            next_done = torch.tensor(terminations).to(device).view(-1)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).reshape((1, -1)).to(device), torch.Tensor(next_done).reshape((1, -1)).to(device)

            global_step += 1
            c_step += 1

            if c_step >= 1023:
                next_done = torch.ones_like(next_done).to(device)

            if first_rollout:
                rollout_return += reward
                if next_done.item() or step == args.num_steps - 1:
                    first_rollout = False
                    records["reward"][iteration - 1] = rollout_return

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if dones[t]:
                    lastgaelam = 0
                    continue
                # if t == args.num_steps - 1:
                #     nextnonterminal = 1.0 - next_done
                #     nextvalues = next_value
                # else:
                #     nextnonterminal = 1.0 - dones[t]
                #     nextvalues = values[t + 1]
                # delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                if t == args.num_steps - 1:
                    nextvalues = next_value
                else:
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, 4))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, 1))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Policy loss
                if iteration > args.num_iterations * -1:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                else:
                    with torch.no_grad():
                        expert_action = object_dict["Kpz"] * b_obs[mb_inds][:, 0] + object_dict["Kiz"] * b_obs[mb_inds][:, 1]
                        expert_action = expert_action.reshape((-1, 1))
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], expert_action)
                    pg_loss = -newlogprob.mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2 * (1-b_dones[mb_inds])).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                for _ in range(2):
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2 * (1-b_dones[mb_inds])).mean()
                    loss = v_loss * args.vf_coef
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # records["reward"][iteration - 1] = rewards.sum().item()
        records["learning_rate"][iteration - 1] = optimizer.param_groups[0]["lr"]
        records["value_loss"][iteration - 1] = v_loss.item()
        records["policy_loss"][iteration - 1] = pg_loss.item()
        records["entropy"][iteration - 1] = entropy_loss.item()

        cos_annealing_scheduler.step()

        if iteration == args.num_iterations:
            agent.eval()
            object_dict["theta"] = object_dict["alpha"] = 0.00028
            states = []
            obses = []
            actions = []
            rewards = []
            values = []
            env = aerodrome.make("wingedcone-v0")
            object = WingedCone2D_RL(object_dict)
            env.add_object(object)
            next_obs, info = env.reset()
            next_obs = torch.Tensor(next_obs).reshape((1, -1)).to(device)
            step = 0
            while True:
                step += 1
                # ALGO LOGIC: action logic
                states.append(env.get_state())
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    obses.append(next_obs.cpu().numpy())
                    actions.append(action.item())
                    values.append(value.item())
                # TRY NOT TO MODIFY: execute the game and log data.
                step_action = {
                    "test": {"Nyc":0.0, "Vc":4590.29, "nn_control":action.item()},
                }
                next_obs, reward, terminations, truncations, infos = env.step(step_action)
                next_done = np.logical_or(terminations, truncations)
                next_obs = torch.Tensor(next_obs).reshape((1, -1)).to(device)
                rewards.append(reward)
                if next_done or step >= 1000:
                    break
            fig, ax = plt.subplots(1, 1)
            # x = np.array([states[i]["pos"][0] for i in range(len(states))])
            x = np.arange(len(states))
            # y = np.array([states[i]["Ny"] for i in range(len(states))])
            # ax.plot(x, y, label="Ny")
            # y = np.array([states[i]["eNy"] for i in range(len(states))])
            # ax.plot(x, y, label="eNy")
            # y = np.array([states[i]["alpha"] for i in range(len(states))])
            # ax.plot(x, y, label="alpha")
            # y = np.array([states[i]["theta"] for i in range(len(states))])
            # ax.plot(x, y, label="theta")
            # y = np.array([states[i]["theta_v"] for i in range(len(states))])
            # ax.plot(x, y, label="theta_v")
            # y = np.array([states[i]["i_eNy"] for i in range(len(states))])
            # ax.plot(x, y, label="i_eNy")
            # y = np.array([states[i]["d_eNy"] for i in range(len(states))])
            # ax.plot(x, y, label="d_eNy")

            # y1 = np.array([states[i]["p_eNy"] for i in range(len(states))])
            # y2 = np.array([states[i]["i_eNy"] for i in range(len(states))])

            # c = object_dict["Kiz"] * y2 + object_dict["Kpz"] * y1
            # ax.plot(x, c, ls='--', label="c")
            # y = np.array([actions[i] for i in range(len(actions))])
            # ax.plot(x, y, ls='--', label="action")

            # ax.plot(x, rewards, label="reward")
            # ax.plot(x, y1, label="p_eNy")
            # ax.plot(x, y2, label="i_eNy")
            y = np.array([states[i]["Ny"] for i in range(len(states))])
            # y1 = np.array([states[i]["eNy"] for i in range(len(states))])
            # y2 = np.array([states[i]["d_eNy"] for i in range(len(states))])
            # r1 = np.clip(1-np.abs(y1), -3.0, 1.0)
            # r2 = np.clip((-y2/y1)-0.5, -0.5, 0.5)
            # r3 = np.array([-1.0 if (i>0 and y2[i]*y2[i-1]<0) else 0.0 for i in range(len(y2))])
            np.save("wingedcone_ppo_0.npy", y)
            ax.plot(x, y, label="Ny")
            # ax.plot(x, r1, label="r1")
            # ax.plot(x, r2, label="r2")
            # ax.plot(x, r3, label="r3")
            ax.plot(x, rewards, label="reward")
            ax.axhline(y=1.05, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
            ax.legend()
            plt.show()

            fig, ax = plt.subplots(1, 1)
            ax.plot(x, values, label="value")
            ax.plot(x, rewards, label="reward")
            ax.legend()
            plt.show()

            fig, ax = plt.subplots(1, 1)
            y0 = np.array([obses[i][0,-3] for i in range(len(obses))])
            y1 = np.array([obses[i][0,-2] for i in range(len(obses))])
            y2 = np.array([obses[i][0,-1] for i in range(len(obses))])
            # y3 = np.array([obses[i][0,-1] for i in range(len(obses))])
            ax.plot(x, y0, label="alpha")
            ax.plot(x, y1, label="q")
            ax.plot(x, y2, label="eNy")
            # ax.plot(x, y3, label="Ny")
            ax.axhline(y=0.0, color='r', linestyle='-.', alpha=0.3)
            ax.legend()
            plt.show()

            fig, ax = plt.subplots(1, 1)
            c = object_dict["Kiz"] * y2 + object_dict["Kpz"] * y1
            ax.plot(x, c, ls='--', label="expert")
            y = np.array([actions[i] for i in range(len(actions))])
            ax.plot(x, y, ls='--', label="real")
            ax.axhline(y=0.0, color='r', linestyle='-.', alpha=0.3)
            ax.legend()
            plt.show()

            fig, ax = plt.subplots(1, 1)
            ax.plot(x, [actions[i] for i in range(len(actions))], label="action")
            ax.legend()
            plt.show()

            print(env.get_state())

    # np.save("wingedcone_ppo_0.npy", records["reward"])
    fig, ax = plt.subplots(1, 1)
    ax.plot(records["reward"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(records["policy_loss"], label="policy loss")
    # ax.plot(records["value_loss"], label="value loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("policy_loss")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(records["value_loss"], label="value loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("value_loss")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(records["entropy"], label="entropy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("entropy")
    ax.legend()
    plt.show()

    torch.save(agent.state_dict(), "models/wingedcone_ppo.pth")

if __name__ == "__main__":
    main()