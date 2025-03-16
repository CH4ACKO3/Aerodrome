# Aerodrome #

Aerodrome is a lightweight library based on the joint programming of C++ and Python, focusing on providing high-performance dynamics simulation, Python interaction interfaces, and code examples for reinforcement learning and traditional control. It primarily targets two types of users: beginners in reinforcement learning research, or researchers who wish to apply reinforcement learning to specific fields (such as flight control) and need custom simulation environments. To ensure good modifiability and readability, every part of Aerodrome avoids the use of complex generic programming or advanced language features, and includes detailed comments in the code.

The main contents and features of Aerodrome include:

- Mimicking the code and interface style of Gymnasium

- An example environment and a migrated version of Gym-CartPole for testing the correctness of reinforcement learning algorithms

- Highly customizable fixed-wing aircraft simulation example code

- A nonlinear aerodynamic model of the Winged-Cone based on publicly available data, along with its dynamics simulation (available in both C++ and Python, including performance comparisons!)

- Traditional three-loop autopilot implementation and Proximal Policy Optimization (PPO) implementation for longitudinal load control of the Winged-Cone

> **Note**  
> Aerodrome is **not** a modular library, therefore it is not suitable for import and use. Aerodrome sacrifices some code duplication to ensure that all code is concise, easy to understand, and convenient to extend. Aerodrome also has its limitations, such as the lack of visualization and experimental result saving functions; if you need specific reinforcement learning algorithm examples, you can refer to the CleanRL library; if you need to implement other aircraft models (or even environments in other fields), you can modify the Aerodrome code or rewrite it entirely!

# Install Aerodrome #

1.Create and activate conda environment

```bash
$ conda create -n aerodrome python==3.9
$ conda activate aerodrome
```

> **Warning**  
> The binary files provided in the source code were compiled on a Windows platform using ```python==3.9```. You may need to recompile them to adapt to your own environment (refer to 4.).

2.Install Pytorch

Running reinforcement learning code requires the use of the ```torch``` library; the source code has been tested on ```torch==2.6.0```, but theoretically, it can run on any ```torch>1.0.0``` and compatible CUDA versions. For example, ```PyTorch 2.6.0``` with ```CUDA 11.8```:

```bash
$ pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

3.Install Aerodrome from source

```bash
$ git clone https://github.com/CH4ACKO3/Aerodrome.git --recursive
$ cd Aerodrome
$ pip install .
```

4.Compiling the C++ Code (Optional)

If you intend to use a custom environment or a Python version that differs from the one used to generate the binary files in the source code (```python==3.9```), you will need to recompile and install the code:

```bash
$ cmake -B build
$ cmake --build build
$ pip install .
```

# Get Started #

Run the example code and observe the longitudinal step response of the Winged-Cone controlled by the three-loop autopilot!

```bash
$ python examples/AircraftControl/examples/AircraftControl/WingedCone_Classic.py
```
