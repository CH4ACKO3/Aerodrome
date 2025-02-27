from aerodrome.core import Env
from aerodrome.registration import register
from aerodrome.simulator.MinimalExample.envs import DerivedEnv

class ExampleEnv(Env):
    def __init__(self):
        self.env = DerivedEnv()
        self.state = 0
        print("Init")

    def step(self, action):
        self.state += 1
        try:
            input_dict = {
                "value": action,
                "state": self.state,
                "metadata": {
                    "description": "input",
                    "source": "cmdline input"
                },
                "other_info": [1, 2, 3]
            }
            result = self.env.step(input_dict)
            print(result)
        except ValueError:
            print("input a valid integer")
        except KeyboardInterrupt:
            print("\nquit")
    
    def reset(self):
        self.state = 0
        self.env.reset()
        print("Reset")
    
    def close(self):
        print("Close")

register("example-v0", "aerodrome.envs.example:ExampleEnv")