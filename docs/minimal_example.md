# 最简环境 #
为了方便用户理解 Aerodrome 的运行逻辑，这里提供一个最简的例子。

使用到的代码：

- C++ 环境： `src/simulator/MinimalExample/envs/MinimalEnv.h`
- Pybind11 绑定： `src/simulator/MinimalExample/envs/MinimalEnv.cpp`
- Python 环境： `python/aerodrome/envs/Minimal.py`
- 交互代码： `examples/MinimalExample/MinimalExample.py`

## C++ 环境 ##

```cpp
class MinimalEnv : public BaseEnv { // 继承自 BaseEnv
public:
    MinimalEnv() : cpp_state(0) {} // 初始化 cpp_state 为 0

    ~MinimalEnv() {} // 析构函数

    py::object reset() override // 重写 BaseEnv 中定义的 reset 方法
    {
        py::dict result; // 创建一个字典来存储要返回的结果
        cpp_state = 0; // 将 cpp_state 重置为 0
        result["cpp_state"] = cpp_state;
        result["info"] = "";
        return result;
    }

    py::object step(const py::object& action) override // step 方法从 Python 端接收动作
    {
        // 在每一步中，环境将 action["value"] 的值加到其 cpp_state 上
        py::dict result;
        if (action.contains("value"))
        {
            try {
                int value = action["value"].cast<int>(); // 将 action["value"] 的值转换为整数
                cpp_state += value; // 将该值加到 cpp_state 上
            } catch (const std::exception &e) {
                result["info"] = std::string("failed to convert value to int: ") + e.what(); // 如果该值不是整数，在结果中添加错误信息
            }
        }
        else
        {
            result["info"] = "error: action must contain 'value'";
        }
        
        result["cpp_state"] = cpp_state;
        if (!result.contains("info"))
        {
            // 如果结果中没有 info 键，添加一个空字符串，以避免 key error
            result["info"] = "";
        }
        return result;
    }

private:
    int cpp_state = 0; // 私有成员变量，用于存储环境的状态
};
```

## Python 环境 ##

```python
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
```

## 交互代码 ##

```python
if __name__ == "__main__":
    env = aerodrome.make("minimal-v0") # 创建 Python 环境
    result = env.reset() # 重置环境
    print(result) # 打印重置后返回的结果

    while True:
        try:
            action = input("Enter an action value: ") # 从用户输入获取动作（一个整数）
        except KeyboardInterrupt:
            env.close()
            break
        
        try:
            action = int(action) # 将用户输入的动作转换为整数
        except ValueError:
            print("Invalid action value")
            continue
        result = env.step(action) # 调用环境 step 方法，并接收返回结果
        print(result) # 打印返回结果

        if result["py_state"] > 10: # 如果 Python 环境内部状态大于 10（大于 10 步），则重置环境
            result = env.reset()
            print(result)
```

## 运行结果 ##

```bash
$ python examples/MinimalExample/MinimalExample.py
Initialize MinimalEnv
Reset MinimalEnv
{'cpp_state': 0, 'info': '', 'py_state': 0}
Enter an action value: 2
{'cpp_state': 2, 'info': '', 'py_state': 1}
Enter an action value: 12
{'cpp_state': 14, 'info': '', 'py_state': 2}
Enter an action value: 18
{'cpp_state': 32, 'info': '', 'py_state': 3}
Enter an action value: Close MinimalEnv
```

## Pybind11 绑定 ##

如果你需要自己实现环境，应当按照如下格式写好绑定代码，或参考 `pybind11` 的 [官方文档](https://pybind11.readthedocs.io/en/stable/advanced/classes.html)。
```cpp
PYBIND11_MODULE(MinimalEnv, m) {
    py::class_<MinimalEnv, BaseEnv>(m, "MinimalEnv")
        .def(py::init<>())
        .def("reset", &MinimalEnv::reset)
        .def("step", &MinimalEnv::step);
}
```
