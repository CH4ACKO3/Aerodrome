# 安装 Aerodrome #

1.创建并激活 Conda 环境

```bash
$ conda create -n aerodrome python==3.9
$ conda activate aerodrome
```

2.安装 Pytorch

运行强化学习代码需要用到 ```torch``` 库；源码在 ```torch==2.6.0``` 上测试运行，但理论上可以在任意 ```torch>1.0.0``` 以及兼容的 CUDA 版本上运行。例如，```PyTorch 2.6.0``` 和 ``` CUDA 11.8 ```：

```bash
$ pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

3.从源代码安装 Aerodrome

```bash
$ git clone https://github.com/CH4ACKO3/Aerodrome.git
$ cd Aerodrome
$ pip install .
```

4.安装依赖库

```bash
$ pip install -r requirements.txt
```