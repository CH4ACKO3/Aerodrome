import os
import sys
pyd_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build/Release'))
sys.path.append(pyd_path)

import DerivedEnv

class PythonEnv:
    def __init__(self):
        self.env = DerivedEnv.Env()

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
                result = self.env.step(input_dict)
                print(result)
            except ValueError:
                print("input a valid integer")

            except KeyboardInterrupt:
                print("\nquit")
                break

if __name__ == "__main__":
    python_env = PythonEnv()
    python_env.interact()
