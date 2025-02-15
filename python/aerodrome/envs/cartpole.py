from aerodrome.core import Env
from aerodrome.registration import register
from aerodrome.simulator.envs.CartPoleEnv import CartPoleEnv

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
    
register("cartpole-v0", "aerodrome.envs.cartpole:CartPole")