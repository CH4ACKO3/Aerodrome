from aerodrome.core import Env
from aerodrome.registration import register
from aerodrome.simulator.MinimalExample.envs.MinimalEnv import MinimalEnv

class Minimal(Env):
    def __init__(self):
        self.env = MinimalEnv()
        self.state = 0
        print("Initialize MinimalEnv")

    def step(self, action):
        self.state += 1
        try:
            input_dict = {
                "value": action,
            }
            result = self.env.step(input_dict)
            result["py_state"] = self.state
            return result
        except ValueError:
            print("input a valid integer")
        except KeyboardInterrupt:
            print("\nquit")
    
    def reset(self):
        self.state = 0
        print("Reset MinimalEnv")
        result = self.env.reset()
        result["py_state"] = self.state
        return result
    
    def close(self):
        print("Close MinimalEnv")

register("minimal-v0", "aerodrome.envs.minimal:Minimal")