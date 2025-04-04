# CartPole 和 DQN #

在 Aerodrome 中，我们提供了一个从 `gymnasium` 复刻的 CartPole 环境，并使用最基础的 DQN 算法进行训练。用户可以在这个简单的环境上试验自己的算法，以验证代码的正确性。

使用到的代码：

- C++ 环境： `src/simulator/CartPole/envs/CartPoleEnv.h`
- Python 环境： `python/aerodrome/envs/CartPole.py`
- 交互代码： `examples/CartPole/CartPole_dqn.py`

## C++ 环境 ##

倒立摆仿真，迁移自 `gymnasium` 的 [CartPole](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py)。<del>其实 gymnasium 也是复制的别人的[代码](http://incompleteideas.net/sutton/book/code/pole.c)</del>

<details>
<summary>点击展开代码</summary>

```cpp title="CartPoleEnv.h"
class CartPoleEnv : public BaseEnv
{
private:
    double gravity = 9.81;
    double masscart = 1.0;
    double masspole = 0.1;
    double total_mass = masspole + masscart;
    double length = 0.5; // half the pole's length
    double polemass_length = masspole * length;
    double force_mag = 10.0;
    double tau = 0.02; // seconds between state updates
    std::string kinematic_integrator = "euler";

    double theta_threshold_radians = 12 * 2 * 3.1415926 / 360; // angle at which to fail the episode
    double x_threshold = 2.4; // distance at which to fail the episode

    double x = 0;
    double theta = 0;
    double x_dot = 0;
    double theta_dot = 0;

    int time_step = 0;
    int max_steps = 200;
    bool steps_beyond_done = false;

public:
    CartPoleEnv() {}

    ~CartPoleEnv() {}

    py::object reset() override
    {
        py::dict result;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.05, 0.05);

        x = dis(gen);
        x_dot = dis(gen);
        theta = dis(gen);
        theta_dot = dis(gen);
        time_step = 0;
        steps_beyond_done = false;

        result["observation"] = py::make_tuple(x, x_dot, theta, theta_dot);
        result["info"] = "";
        return result;
    }

    py::object step(const py::object& input_dict) override
    {
        py::dict result;
        if (!input_dict.contains("action"))
        {
            result["info"] = "input_dict does not contain 'action'";
            return result;
        }

        int action;
        try
        {
            action = input_dict["action"].cast<int>();
        }
        catch (const std::exception &e)
        {
            result["info"] = std::string("failed to convert action to int: ") + e.what();
            return result;
        }

        if (action != 0 && action != 1)
        {
            result["info"] = "action must be either 0 or 1";
            return result;
        }

        double force = action * force_mag;
        double costheta = cos(theta);
        double sintheta = sin(theta);

        // For the interested reader:
        // https://coneural.org/florian/papers/05_cart_pole.pdf
        double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        double theta_acc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        double x_acc = temp - polemass_length * theta_acc * costheta / total_mass;

        if (kinematic_integrator == "euler")
        {
            x = x + tau * x_dot;
            x_dot = x_dot + tau * x_acc;
            theta = theta + tau * theta_dot;
            theta_dot = theta_dot + tau * theta_acc;
        }
        else if (kinematic_integrator == "semi-implicit-euler")
        {
            x_dot = x_dot + tau * x_acc;
            x = x + tau * x_dot;
            theta_dot = theta_dot + tau * theta_acc;
            theta = theta + tau * theta_dot;
        }
        else
        {
            result["info"] = "unknown kinematic integrator";
            return result;
        }

        time_step ++;
        bool terminated = ((x < -x_threshold) || (x > x_threshold) || (theta < -theta_threshold_radians) || (theta > theta_threshold_radians));
        bool truncated = (time_step >= max_steps);

        double reward;
        if (!terminated && !truncated)
        {
            reward = 1.0;
        }
        else if (!steps_beyond_done)
        {
            if (terminated)
            {
                reward = 0.0;
            }
            else if (truncated)
            {
                reward = 1.0;
            }
            steps_beyond_done = true;
        }
        else
        {
            reward = 0.0;
            result["info"] = "You are calling 'step()' even though this environment has already returned terminated = True. "
                              "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior.";
        }

        result["observation"] = py::make_tuple(x, x_dot, theta, theta_dot);
        result["reward"] = reward;
        result["terminated"] = terminated;
        result["truncated"] = truncated;

        if (!result.contains("info"))
        {
            result["info"] = "";
        }

        return result;
    }
};
```
</details>


## Python 环境 ##

非常简单的 Python 环境，把 C++ 环境包装了一下，把返回值处理成 `gymnasium` 的格式。

```py title="CartPole.py"
class CartPole(Env):
    def __init__(self):
        self.env = CartPoleEnv()

    def reset(self):
        output_dict = self.env.reset()
        return output_dict["observation"], output_dict["info"]

    def step(self, action: int):
        input_dict = {
            "action": action
        }

        output_dict = self.env.step(input_dict)
        return output_dict["observation"], output_dict["reward"], output_dict["terminated"], output_dict["truncated"], output_dict["info"]
```

## DQN ##

这里实现的 DQN 代码来自 `CleanRL` 的 [DQN](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py)，进行了一定的简化，去掉了结果储存等部分。

<details>
<summary>点击展开代码</summary>

```py title="CartPole_dqn.py"
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
```
</details>


## 运行结果 ##

仅作演示用，未经过仔细调参，可以看到有一定的训练效果。

<figure markdown="span">
  ![Image title](./assets/cartpole.png)
</figure>