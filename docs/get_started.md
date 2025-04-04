# 安装 Aerodrome #

1.创建并激活 Conda 环境

```bash
$ conda create -n aerodrome python==3.9
$ conda activate aerodrome
```

> Aerodrome 在 ```py>=3.9.0,<3.12``` 上测试，但源码中包含的二进制文件（python/simulator 下的 C++ 编译出的文件）是在 ```py==3.9``` 环境下生成的；如果你要使用其它 python 版本或自己实现的环境，则需要重新编译并**保证编译和运行时的 python 版本一致**。

2.安装 Pytorch

运行强化学习代码需要用到 `torch` 库；源码在 `torch==2.6.0` 上测试运行，但理论上可以在任意 `torch>1.0.0` 以及兼容的 CUDA 版本上运行。例如，`PyTorch 2.6.0` 和 ` CUDA 11.8 `：

```bash
$ pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

3.从源代码安装 Aerodrome

```bash
$ git clone https://github.com/CH4ACKO3/Aerodrome.git --recursive
$ cd Aerodrome
$ pip install .
```

4.编译 C++ 部分代码（可选）

如果你要使用自己实现的环境，或使用的 python 版本与源码中二进制文件（`python==3.9`）不一致，则需要重新编译并安装：

```bash
$ cmake -B build
$ cmake --build build
$ pip install .
```

> 编译时的平台和 python 版本会在二进制文件名中显示；例如，`*.cp39-win_amd64.pyd` 表示文件是在 Windows 平台、64位架构下，`python==3.9` 环境中编译的。