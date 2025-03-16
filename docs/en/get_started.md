# Install Aerodrome #

1.Create and activate conda environment

```bash
$ conda create -n aerodrome python==3.9
$ conda activate aerodrome
```

> Aerodrome has been tested on ```python>=3.9.0,<3.12```, but the binary files included in the source code (C++ compiled files under python/simulator) were generated in a ```python==3.9``` environment. If you intend to use a different Python version or your own custom environment, you will need to recompile the binaries and **ensure that the Python versions used for compilation and runtime are consistent**.

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

4.Compiling the C++ Code

If you intend to use a custom environment or a Python version that differs from the one used to generate the binary files in the source code (```python==3.9```), you will need to recompile and install the code:

```bash
$ cmake -B build
$ cmake --build build
$ pip install .
```

The platform and Python version used during compilation will be reflected in the binary file names. For example, a file named *.cp39-win_amd64.pyd indicates that it was compiled on a Windows platform with a 64-bit architecture and in a ```python==3.9``` environment.