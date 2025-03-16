# 项目结构 #

Aerodrome 由 C++ 和 Python 两部分组成；其中 C++ 函数/类需要先通过 `pybind11` 编译为 `.pyd` 或 `.so` 文件，再被 Python 代码调用。

```
./Aerodrome
├─docs                              # 项目文档
├─examples                          # 示例
│  ├─AircraftControl       
│  │  ├─ StepResponse.py            # 传统控制和强化学习控制的阶跃响应比较
│  |  ├─ WingedCone_Classic.py      # 传统控制方法(直接调用 C++ 代码)
│  |  ├─ WingedCone_PPO.py          # 强化学习控制(C++ 仿真/ Python 交互环境)
│  |  └─ WingedCone_TimeCompare.py  # C++ 和 Python 仿真运行速度比较
│  ├─CartPole                       # DQN 实现的 CartPole 控制
│  └─MinimalExample                 # 最简示例环境
├─include
│  └─pybind11
├─models                            # 预训练模型
├─python                            # Python 源码
│  ├─aerodrome 
│  │  ├─envs                        # 交互环境，处理用户和C++仿真之间的信息交互
│  │  └─simulator                   # 编译好的 C++ 代码会自动存放到此处
│  │      ├─CanonicalAircraftEnv 
│  │      │  ├─envs                 # 环境类(通常情况下环境和实体是作为不同类编写的)
│  │      │  └─objects              # 实体类
│  │      ├─CartPole
│  │      │  └─envs
│  │      ├─Core
│  │      │  ├─envs
│  │      │  └─objects
│  │      └─MinimalExample
│  │          └─envs
│  └─aerodrome.egg-info
└─src                              # C++ 源码
   └─simulator
       ├─CanonicalAircraftEnv
       │  ├─envs
       │  └─objects
       ├─CartPole
       │  ├─envs
       │  └─objects
       ├─Core
       │  ├─envs
       │  └─objects
       └─MinimalExample
           ├─envs
           └─objects
```