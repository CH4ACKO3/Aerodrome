# Install Aerodrome #

1.Create and activate conda environment

```bash
$ conda create -n aerodrome python==3.9
$ conda activate aerodrome
```

2.Install Pytorch

Running reinforcement learning code requires the use of the ```torch``` library; the source code has been tested on ```torch==2.6.0```, but theoretically, it can run on any ```torch>1.0.0``` and compatible CUDA versions. For example, ```PyTorch 2.6.0``` with ```CUDA 11.8```:

```bash
$ pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

3.Install Aerodrome from source

```bash
$ git clone https://github.com/CH4ACKO3/Aerodrome.git
$ cd Aerodrome
$ pip install .
```

4.Install requirements

```bash
$ pip install -r requirements.txt
```