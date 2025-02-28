import random
import numpy as np
import argparse
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import aerodrome

class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.full = False
        self.pointer = 0
        self.obs = np.zeros((buffer_size, 4), dtype=np.float32)
        self.act = np.zeros((buffer_size, 1), dtype=np.float32)
        self.reward = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, 4), dtype=np.float32)
        self.terminated = np.zeros((buffer_size, 1), dtype=np.float32)
        self.truncated = np.zeros((buffer_size, 1), dtype=np.float32)

    def push(self, obs, act, reward, next_obs, terminated, truncated):
        self.obs[self.pointer] = obs
        self.act[self.pointer] = act
        self.reward[self.pointer] = reward
        self.next_obs[self.pointer] = next_obs
        self.terminated[self.pointer] = terminated
        self.truncated[self.pointer] = truncated
        self.pointer = (self.pointer + 1) % self.buffer_size
        if not self.full and self.pointer == 0:
            self.full = True

    def sample(self, batch_size: int):
        if self.full:
            indices = np.random.randint(0, self.buffer_size, size=batch_size)
        else:
            indices = np.random.randint(0, self.pointer, size=batch_size)
        
        data = {
            "obs": self.obs[indices],
            "act": self.act[indices],
            "reward": self.reward[indices],
            "next_obs": self.next_obs[indices],
            "terminated": self.terminated[indices],
            "truncated": self.truncated[indices]
        }
        return data

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.network(x)
    
def linear_schedule(start_e, end_e, duration, t):
    return start_e + (end_e - start_e) * min(t / duration, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed of the experiment")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="the target network update rate")
    parser.add_argument("--target_network_frequency", type=int, default=1_000,
                        help="the frequency of target network update")
    parser.add_argument("--start_e", type=float, default=1.0,
                        help="starting epsilon")
    parser.add_argument("--end_e", type=float, default=0.01,
                        help="ending epsilon")
    parser.add_argument("--exploration_fraction", type=float, default=0.5,
                        help="the timesteps it takes to update the target network")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000,
                        help="total number of timesteps")
    parser.add_argument("--buffer_size", type=int, default=100_000,
                        help="buffer size")
    parser.add_argument("--learning_starts", type=int, default=10_000,
                        help="timestep to start learning")
    parser.add_argument("--train_frequency", type=int, default=10,
                        help="the frequency of training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the experiment on")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    print(f"Using device: {device}")

    q_network = QNetwork().to(device)
    target_network = QNetwork().to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    cos_annealing_scheduler = CosineAnnealingLR(optimizer, T_max=args.total_timesteps, eta_min=args.learning_rate * 0.01)

    replay_buffer = ReplayBuffer(args.buffer_size)
    
    env = aerodrome.make("cartpole-v0")
    obs, info = env.reset()
    obs = np.array(obs, dtype=np.float32)
    print(obs, info)

    episode_length = []
    current_step = 0
    for step in tqdm(range(args.total_timesteps)):
        current_step += 1
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, step)
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                q_values = q_network(torch.from_numpy(obs).unsqueeze(0).to(device))
                action = torch.argmax(q_values, dim=1).cpu().numpy().item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = np.array(next_obs, dtype=np.float32)
        
        real_next_obs = deepcopy(next_obs)
        replay_buffer.push(obs, action, reward, real_next_obs, terminated, truncated)

        obs = next_obs

        if step > args.learning_starts:
            if step % args.train_frequency == 0:
                batch = replay_buffer.sample(args.batch_size)
                batch["obs"] = torch.from_numpy(batch["obs"]).to(device)
                batch["act"] = torch.from_numpy(batch["act"]).to(device)
                batch["reward"] = torch.from_numpy(batch["reward"]).to(device)
                batch["next_obs"] = torch.from_numpy(batch["next_obs"]).to(device)
                batch["terminated"] = torch.from_numpy(batch["terminated"]).to(device)
                batch["truncated"] = torch.from_numpy(batch["truncated"]).to(device)

                with torch.no_grad():
                    target_max, _ = target_network(batch["next_obs"]).max(dim=1, keepdim=True)
                    td_target = batch["reward"] + args.gamma * (1 - batch["terminated"]) * target_max - 10.0 * batch["terminated"]

                old_val = q_network(batch["obs"])
                old_val = old_val.gather(1, batch["act"].long())

                loss = F.mse_loss(td_target, old_val)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if step % args.target_network_frequency == 0:
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        cos_annealing_scheduler.step()

        if terminated or truncated:
            obs, info = env.reset()
            obs = np.array(obs, dtype=np.float32)
            episode_length.append((step, current_step))
            current_step = 0

    episode_length = np.array(episode_length)
    fig, ax = plt.subplots()
    ax.plot(episode_length[:, 0], episode_length[:, 1], alpha=0.5, c="skyblue")
    moving_average = np.convolve(episode_length[:, 1], np.ones(100) / 100, mode='valid')
    ax.plot(episode_length[99:, 0], moving_average, c="royalblue")
    ax.set_xlabel("steps")
    ax.set_ylabel("cumulative reward")
    plt.show()

if __name__ == "__main__":
    main()