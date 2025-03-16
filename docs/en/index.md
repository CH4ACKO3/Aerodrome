# Aerodrome - Overview

Aerodrome is a lightweight library based on the joint programming of C++ and Python, focusing on providing high-performance dynamics simulation, Python interaction interfaces, and code examples for reinforcement learning and traditional control. It primarily targets two types of users: beginners in reinforcement learning research, or researchers who wish to apply reinforcement learning to specific fields (such as flight control) and need custom simulation environments. To ensure good modifiability and readability, every part of Aerodrome avoids the use of complex generic programming or advanced language features, and includes detailed comments in the code.

The main contents and features of Aerodrome include:

- Mimicking the code and interface style of Gym

- An example environment and a migrated version of Gym-CartPole for testing the correctness of reinforcement learning algorithms

- Highly customizable fixed-wing aircraft simulation example code

- A nonlinear aerodynamic model of the Winged-Cone based on publicly available data, along with its dynamics simulation (available in both C++ and Python, including performance comparisons!)

- Traditional three-loop autopilot implementation and Proximal Policy Optimization (PPO) implementation for longitudinal load control of the Winged-Cone

> **Note**  
> Aerodrome is **not** a modular library, therefore it is not suitable for import and use. Aerodrome sacrifices some code duplication to ensure that all code is concise, easy to understand, and convenient to extend. Aerodrome also has its limitations, such as the lack of visualization and experimental result saving functions; if you need specific reinforcement learning algorithm examples, you can refer to the CleanRL library; if you need to implement other aircraft models (or even environments in other fields), you can modify the Aerodrome code or rewrite it entirely!