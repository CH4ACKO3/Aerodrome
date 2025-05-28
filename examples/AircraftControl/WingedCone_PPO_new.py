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

state_dim = 3

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, 1))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, evaluate=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            if evaluate:
                probs = Normal(action_mean, action_std * 1e-6)
            action = probs.sample()
            action = torch.clamp(action, -1, 1)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed of the experiment")
    parser.add_argument("--num_steps", type=int, default=1024,
                        help="the number of steps to run per policy rollout")
    parser.add_argument("--total_timesteps", type=int, default=50_000,
                        help="the number of iterations")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.99,
                        help="lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type=int, default=1,
                        help="the number of mini-batches")
    parser.add_argument("--update_epochs", type=int, default=1,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm_adv", type=bool, default=False,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip_coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip_vloss", type=bool, default=True,
                        help="Toggles whether or not to use a clipped loss for the value function")
    parser.add_argument("--ent_coef", type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="the target KL divergence threshold")
                        
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the experiment on")
    args = parser.parse_args()
    
    args.batch_size = args.num_steps
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    agent = Agent().to(device)
    # print named parameters
    # for name, param in agent.named_parameters():
    #     print(name, param.shape, param.requires_grad, param.mean(), param.std())
    # exit()
    # agent.load_state_dict(torch.load("models/wingedcone_ppo_new.pth"))
    agent.load_state_dict(torch.load("pretrain_delta.pth"))
    agent.actor_logstd.data = torch.ones_like(agent.actor_logstd.data) * np.log(0.01)
    agent.train()
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iterations, eta_min=args.learning_rate*0.1)
    
    records = {
        "reward": np.zeros(args.num_iterations),
        "learning_rate": np.zeros(args.num_iterations),
        "value_loss": np.zeros(args.num_iterations),
        "policy_loss": np.zeros(args.num_iterations),
        "entropy": np.zeros(args.num_iterations),
    }

    # TRY NOT TO MODIFY: start the game
    global_step = 0

    env = aerodrome.make("wingedcone-v0")
    object = WingedCone2D_RL(object_dict)
    env.add_object(object)
    
    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).reshape((1, -1)).to(device)
    next_done = torch.zeros((1, 1)).to(device)

    for iteration in tqdm(range(1, args.num_iterations + 1)):
        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, state_dim)).to(device)
        actions = torch.zeros((args.num_steps, 1)).to(device)
        logprobs = torch.zeros((args.num_steps, 1)).to(device)
        rewards = torch.zeros((args.num_steps, 1)).to(device)
        dones = torch.zeros((args.num_steps, 1)).to(device)
        terminations = torch.zeros((args.num_steps, 1)).to(device)
        values = torch.zeros((args.num_steps, 1)).to(device)

        # Annealing the rate if instructed to do so.
        # frac = 1.0 - (iteration - 1.0) / args.num_iterations
        # lrnow = frac * args.learning_rate
        # optimizer.param_groups[0]["lr"] = lrnow

        A = 0.0
        object_dict["theta"] = object_dict["alpha"] = A
        Nyc = 1.0

        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            if next_done.item():
                # A = 0.00028
                A = np.random.uniform(-0.2, 0.2)/57.3
                object_dict["theta"] = object_dict["alpha"] = A
                Nyc = np.random.uniform(-10.0, 10.0)
                # Nyc = 0.0

                env = aerodrome.make("wingedcone-v0")
                object = WingedCone2D_RL(object_dict)
                env.add_object(object)

                next_obs, _ = env.reset()
                next_obs = torch.Tensor(next_obs).reshape((1, -1)).to(device)
                next_done = torch.zeros((1, 1)).to(device)
                continue

            # TRY NOT TO MODIFY: execute the game and log data.
            step_action = {
                "test": {"Nyc":Nyc, "Vc":4590.29, "nn_control":action.item()},
            }
            for _ in range(10):
                next_obs, reward, termination, truncation, infos = env.step(step_action)
                next_done = np.logical_or(termination, truncation)
                rewards[step] += reward*0.1
                terminations[step] = termination.item()

            next_obs, next_done = torch.Tensor(next_obs).reshape((1, -1)).to(device), torch.Tensor(next_done).to(device)

        # if iteration/args.num_iterations > 0.5:
        #     fig, ax = plt.subplots()
        #     ax.plot(obs[:,-1].cpu().numpy())
        #     plt.show()

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if dones[t].item():
                    lastgaelam = 0
                    continue
                if t == args.num_steps - 1:
                    nextvalues = next_value
                else:
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * (1 - terminations[t]) - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * lastgaelam
            returns = advantages + values

        if iteration == args.num_iterations:
            obs_ = obs.cpu().numpy()
            rewards_ = rewards.cpu().numpy()
            actions_ = actions.cpu().numpy()
            returns_ = returns.cpu().numpy()
            values_ = values.detach().cpu().numpy()
            advantages_ = advantages.cpu().numpy()
            dones_ = dones.cpu().numpy()

            fig, ax = plt.subplots(2, 1)
            ax[0].plot(np.array([obs_[i][1] for i in range(len(obs_))]), label="eNy")
            ax[0].plot(np.array([rewards_[i][0] for i in range(len(rewards_))]), label="reward")
            ax[0].legend()
            ax[1].plot(np.array([actions_[i][0] for i in range(len(actions_))]), label="action")
            ax[1].plot(np.array([dones_[i][0] for i in range(len(dones_))]), label="done")
            ax[1].legend()
            for i in range(len(dones_)):
                if dones_[i][0]:
                    ax[1].axvline(i, color="grey", linestyle="--", alpha=0.3)
            plt.show()

            fig, ax = plt.subplots(2, 1)
            ax[0].plot(np.array([obs_[i][1] for i in range(len(obs_))]), label="eNy")
            ax[0].plot(np.array([rewards_[i][0] for i in range(len(rewards_))]), label="reward")
            ax[0].legend()
            ax[1].plot(np.array([values_[i] for i in range(len(values_))]), label="value")
            ax[1].plot(np.array([returns_[i] for i in range(len(returns_))]), label="return")
            ax[1].plot(np.array([advantages_[i] for i in range(len(advantages_))]), label="advantage")
            for i in range(len(dones_)):
                if dones_[i][0]:
                    ax[1].axvline(i, color="grey", linestyle="--", alpha=0.3)
            ax[1].legend()
            plt.show()
        
        # flatten the batch
        b_obs = obs.reshape((-1, state_dim))
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_dones = b_dones[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_advantages = mb_advantages * (1 - mb_dones)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

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
                    v_loss_max = v_loss_max * (1 - mb_dones)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2 * (1 - mb_dones)).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss * (iteration > args.num_iterations*0.1) - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
            
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        records["reward"][iteration-1] = rewards.sum().item()
        records["learning_rate"][iteration-1] = optimizer.param_groups[0]["lr"]
        records["value_loss"][iteration-1] = v_loss.item()
        records["policy_loss"][iteration-1] = pg_loss.item()
        records["entropy"][iteration-1] = agent.actor_logstd.data.mean().item()

        scheduler.step()

    torch.save(agent.state_dict(), "models/wingedcone_ppo_new.pth")
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(np.array(records["reward"]))
    ax[0, 0].set_title("Reward")
    ax[0, 1].plot(np.array(records["learning_rate"]))
    ax[0, 1].set_title("Learning Rate")
    ax[1, 0].plot(np.array(records["value_loss"]))
    ax[1, 0].set_title("Value Loss")
    ax[1, 1].plot(np.array(records["policy_loss"]), label="policy loss")
    ax[1, 1].set_title("Policy Loss")
    ax[1, 1].plot(np.array(records["entropy"]), label="entropy")
    ax[1, 1].set_title("Entropy")
    ax[1, 1].legend()
    plt.show()

    # eval
    agent.eval()
    states = []
    obs = []
    actions = []
    logprobs = []
    rewards = []
    dones = []
    values = []
    
    env = aerodrome.make("wingedcone-v0")
    A = 0.0
    object_dict["theta"] = object_dict["alpha"] = A
    Nyc = 0.0
    object = WingedCone2D_RL(object_dict)
    env.add_object(object)

    next_obs, _ = env.reset()
    next_obs = torch.Tensor(next_obs).reshape((1, -1)).to(device)
    next_done = torch.zeros((1, 1)).to(device)

    while len(states) < 1024:
        states.append(env.get_state())
        obs.append(next_obs.cpu().numpy())
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs, evaluate=True)
            values.append(value.item())
            actions.append(action.item())
            logprobs.append(logprob.item())

        step_action = {
            "test": {"Nyc":1.0, "Vc":4590.29, "nn_control":action.item()},
        }
        # _r = 0.0
        # for _ in range(10):
        #     next_obs, reward, terminations, truncations, infos = env.step(step_action)
        #     next_done = np.logical_or(terminations, truncations)
        #     _r += reward
        # rewards.append(_r*0.1)
        next_obs, reward, terminations, truncations, infos = env.step(step_action)
        next_done = np.logical_or(terminations, truncations)
        rewards.append(reward)
        next_obs, next_done = torch.Tensor(next_obs).reshape((1, -1)).to(device), torch.Tensor(next_done).to(device)
    
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(np.array(rewards), label="reward")
    ax[0].plot(np.array(values), label="value")
    ax[0].plot(np.array([obs[i][0,1] for i in range(len(obs))]), label="eNy")
    ax[1].plot(np.array(actions), label="action")
    ax[1].plot(np.array(logprobs), label="logprob")
    ax[0].legend()
    ax[1].legend()
    plt.show()

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.array(actions), label="action")
    # ax[0].plot(np.array(values), label="value")
    # ax[1].plot(np.array(actions), label="action")
    # ax[1].plot(np.array(logprobs), label="logprob")
    ax[1].plot(np.array([obs[i][0,0] for i in range(len(obs))]), label="Nyc")
    ax[1].plot(np.array([obs[i][0,1] for i in range(len(obs))]), label="eNy")
    ax[1].plot(np.array([obs[i][0,2] for i in range(len(obs))]), label="d_eNy")
    ax[0].legend()
    ax[1].legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.array([states[i]["Ny"] for i in range(len(states))]), label="Ny")
    ax.plot(np.array([states[i]["eNy"] for i in range(len(states))]), label="eNy")
    ax.axhline(0.0, color="black", linestyle="--")
    ax.legend()
    plt.show()

    np.save("wingedcone_ppo_0.npy", np.array([states[i]["Ny"] for i in range(len(states))]))
    print(env.get_state())
