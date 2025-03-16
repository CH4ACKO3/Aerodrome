# Project Structure #

Aerodrome consists of both C++ and Python components. The C++ functions/classes need to be compiled into `.pyd` files using `pybind11` first, after which they can be called by the Python code.

```
./Aerodrome
├─docs                              # Project Documentation
├─examples                          # Examples
│  ├─AircraftControl       
│  │  ├─ StepResponse.py            # Comparison of Step Response Between Traditional Control and Reinforcement Learning Control
│  |  ├─ WingedCone_Classic.py      # Traditional Control Method (Directly Calling C++ Code)
│  |  ├─ WingedCone_PPO.py          # Reinforcement Learning Control (C++ Simulation / Python Interaction Environment)
│  |  └─ WingedCone_TimeCompare.py  # Comparison of Simulation Execution Speed Between C++ and Python
│  ├─CartPole                       # DQN Implementation for CartPole Control
│  └─MinimalExample                 # Minimal Example Environment
├─include
│  └─pybind11
├─models                            # Pre-trained Models
├─python                            # Python Source Code
│  ├─aerodrome 
│  │  ├─envs                        # Interaction Environment, Handling Information Exchange Between User and C++ Simulation
│  │  └─simulator                   # Compiled C++ Code Will Be Automatically Stored Here
│  │      ├─CanonicalAircraftEnv 
│  │      │  ├─envs                 # Environment Class (Typically, Environment and Entity Are Written as Separate Classes)
│  │      │  └─objects              # Entity Class
│  │      ├─CartPole
│  │      │  └─envs
│  │      ├─Core
│  │      │  ├─envs
│  │      │  └─objects
│  │      └─MinimalExample
│  │          └─envs
│  └─aerodrome.egg-info
└─src                              # C++ Source Code
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