# Python 3.14t 与依赖升级

三平台最新实测见 [构建检查报告](platform-builds.md)：Linux x86_64 和 macOS ARM64 的 3.14t 安装运行通过；Windows 原生打包通过，运行仍受 JAXlib cp314t wheel 缺失限制。

本项目解释器改为 CPython 3.14.7t，JAX/JAXlib 升级为 0.11.1。依赖通过 PyPI 最新稳定版元数据核对，并由 uv 重新解析、锁定。未采用预发布版本。

首次升级时完成 50 项测试及 5 个示例回归；当前完整测试与新增示例见[验证记录](validation.md)。运行时报告在 `artifacts/runtime-314t.json`，现在也检查可选 python-control 的导入和实际响应计算。

| 项目 | 版本 |
|---|---|
| Python | 3.14.7，free-threaded 构建 |
| JAX / JAXlib | 0.11.1 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| python-control（可选） | 0.10.2 |
| Matplotlib（control 依赖） | 3.11.1 |
| pytest | 9.1.1 |
| Hatchling 构建依赖最低版本 | 1.32.0 |
| 本工作区 uv | 0.12.13 |

NumPy、SciPy、pytest 原来已经处于上述最新稳定版，因此没有虚构版本升级。
`.python-version` 指定 `3.14.7t`；`requires-python` 限定 3.14 系列。Python 包版本元数据不能表达是否为 free-threaded 构建，所以另设运行时检查。

## Windows 主机上的运行方式

PyPI 当前 jaxlib 0.11.1 提供 Linux x86_64 的 cp314t wheel，但没有 Windows 原生 cp314t wheel。Windows 普通 cp314 wheel 与 free-threaded ABI 不兼容，不能改名混装。
因此本机使用已有的 Ubuntu WSL，源码仍位于 Windows 工作目录，运行环境和下载缓存放在同一工作区的 `work/`。

另外核对了 PyPI jaxlib 全部历史发布文件，未找到 Windows 的 cp313t/cp314t wheel；降级 JAX 没有现成安装包可解决此组合。普通 Windows CPython 3.14 有 JAXlib 0.11.1 wheel，但它使用 GIL，不满足当前 3.14t 目标。

在项目目录的 PowerShell 执行：

```powershell
./scripts/run-wsl.ps1 sync
./scripts/run-wsl.ps1 check
./scripts/run-wsl.ps1 test
./scripts/run-wsl.ps1 examples
```

这些入口面向当前 `outputs/aerodrome-ngc` 本地原型布局，自动定位工作区中的 uv、Python 与虚拟环境。
没有修改系统默认 Python、全局 uv、Ubuntu 配置或 Windows 旧环境。

## 独立 Linux checkout

安装当前版本的 uv 后，在项目目录运行：

```shell
uv sync --locked --extra test --extra control
uv run --locked --extra control python scripts/check_runtime.py
uv run --locked --extra test --extra control python -m pytest -q
uv run --locked python examples/compiled_modules.py
```

解释器读取 `.python-version`；uv 会按需要下载对应 free-threaded Python。日常运行使用 locked，避免依赖自动漂移。
有意更新版本时执行 `uv lock --upgrade`，然后再次检查运行时、完整测试与示例。

## 验证确实无 GIL

`scripts/check_runtime.py` 检查：

1. CPython 版本为 3.14，构建配置 `Py_GIL_DISABLED=1`。
2. 解释器启动时 `sys._is_gil_enabled()` 为 False。
3. 导入 NumPy、SciPy、JAXlib、JAX 后仍为 False。
4. 真正完成一次 JIT 数值计算之后仍为 False。
5. 安装 control extra 时，导入 python-control 并执行 TF→SS、离散化和阶跃响应后仍为 False。

不能只看解释器文件名，也不能通过强制 `PYTHON_GIL=0` 掩盖不支持 free-threading 的扩展。
项目不会强制关闭 GIL；运行环境不符合要求时检查脚本和测试会明确失败。
这说明测试环境的实际状态，不代表所有第三方 MATLAB/HDF5 扩展都已经通过兼容性验证。

依赖文件分工：`pyproject.toml` 是兼容范围，`uv.lock` 是可复现的解析结果和包校验值，`requirements-tested.txt` 是本次 Linux 环境的版本快照；uv 本身与 Python 解释器不属于 pip 包锁文件。

依据：[Python 3.14.7](https://www.python.org/downloads/release/python-3147/)、[free-threading 官方说明](https://docs.python.org/3/howto/free-threading-python.html)、[JAXlib PyPI 发布文件](https://pypi.org/project/jaxlib/0.11.1/#files)。
