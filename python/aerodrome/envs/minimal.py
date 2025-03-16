from aerodrome.core import Env
from aerodrome.registration import register # 导入 register 函数，用于注册环境
from aerodrome.simulator.MinimalExample.envs.MinimalEnv import MinimalEnv # 从编译好的 C++ 二进制文件导入 MinimalEnv 类

class Minimal(Env):
    def __init__(self):
        self.env = MinimalEnv() # 初始化 C++ 环境
        self.state = 0 # 初始化 Python 环境内部状态
        print("Initialize MinimalEnv")

    def step(self, action):
        self.state += 1 # 在每一步中，Python 环境内部状态加 1
        try:
            input_dict = {
                "value": action, # 将 Python 环境接收到的整数转换为动作字典
            }
            result = self.env.step(input_dict) # 调用 C++ 环境的 step 方法，并接收返回结果
            result["py_state"] = self.state # 将 Python 环境内部状态添加到返回结果中
            return result
        except ValueError:
            print("input a valid integer")
        except KeyboardInterrupt:
            print("\nquit")
    
    def reset(self):
        self.state = 0 # 重置 Python 环境内部状态
        print("Reset MinimalEnv")
        result = self.env.reset() # 调用 C++ 环境的 reset 方法，并接收返回结果
        result["py_state"] = self.state # 将 Python 环境内部状态添加到返回结果中
        return result
    
    def close(self):
        print("Close MinimalEnv") # 在有些情况下，可能需要在关闭环境时进行一些清理工作

register("minimal-v0", "aerodrome.envs.Minimal:Minimal") # 注册环境