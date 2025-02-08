import os
import sys

pyd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/Release'))
sys.path.append(pyd_path)

import simulator

class PythonEnv:
    def __init__(self):

        self.env = simulator.SimulatorEnv()


    def interact(self):
        while True:
            try:
                user_input = int(input("input: "))
                input_dict = {
                    "value": user_input,
                    "metadata": {
                        "description": "input",
                        "source": "cmdline input"
                    },
                    "other_info": [1, 2, 3]
                }
                result_dict = self.env.process_input(input_dict)
                print("result: ", result_dict)
            except ValueError:
                print("input a valid integer")

            except KeyboardInterrupt:
                print("\nquit")
                break

if __name__ == "__main__":
    python_env = PythonEnv()
    python_env.interact()
