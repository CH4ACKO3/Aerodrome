# 三平台构建检查（2026-09-11）

使用相同依赖锁文件、uv 0.12.13、CPython 3.14.7t，以及全部 extras（test/control/hydra/gym），分别检查分发包构建、依赖安装、实际 wheel 导入和测试。

| 平台 | sdist + wheel | 3.14t 依赖安装 | 安装包隔离检查 | 完整测试 |
|---|---|---|---|---|
| Windows x86_64 原生 | 通过 | 阻断：没有 JAXlib cp314t wheel | 无法运行 | 未执行 |
| Linux x86_64 / Ubuntu WSL2 | 通过 | 通过 | 通过 | 211 passed，138.41 秒 |
| macOS 26.4 / Apple Silicon ARM64 | 通过 | 通过 | 通过 | 211 passed，69.82 秒 |

Linux 和 macOS 均在加载 NumPy、SciPy、JAX、Gymnasium、Hydra、python-control 并执行实际 JIT 后保持 GIL 关闭。两者均使用 CPU；没有验证 CUDA、Metal、Intel Mac 或 Linux ARM64。测试耗时包含不同机器的启动和编译成本，不作为平台性能对比。36 条 warning 均为已有 python-control 对 NumPy shape 赋值的弃用提示。

## 实际检查范围

macOS 使用局域网 Mac `192.168.31.25`，SSH 以已有密钥登录 `ch4acko3`。构建、解释器、依赖环境位于 `/tmp/aerodrome-build.mVm4DFtH/`，结果已复制回本项目。该临时目录保留用于复验。

Linux 和 Mac 使用独立的新环境，先按 uv.lock 安装全部依赖，再强制安装实际构建的 wheel。`scripts/check_distribution.py` 通过 `python -I` 运行并校验模块位于环境 site-packages，避免源码目录和 editable 安装掩盖打包缺失。检查了包内配置加载、World JIT step、F-16 气动表完整性，以及两个配置 CLI 的帮助入口。

完整 pytest 回归使用源码测试套件及项目测试路径设置；它与安装包隔离检查分开记录，不将源码测试冒充 wheel-only 测试。

## 打包修复

发现旧 sdist 会包含 `artifacts/` 中的运行记录。本轮在 Hatch 构建配置中排除了 artifacts、dist、虚拟环境和缓存；重建并检查三个平台的 sdist，确认没有这些生成内容。

三个 wheel 均为 `py3-none-any`，内部全部 **88 个文件内容一致**，包括 F-16 NPZ/manifest/LICENSE、YAML、Gymnasium 和 Hydra 适配器。Linux 与 Mac wheel 的 SHA256 一致；Windows ZIP 容器字节有所不同，不能宣称三个压缩文件逐字节一致。包自身不含平台本地扩展，运行平台限制来自 JAXlib 等依赖。

## Windows 阻断原因

在真实 Windows 3.14.7t 解释器下执行 `uv sync --locked --all-extras`，JAXlib 0.11.1 无匹配分发包。PyPI 发布文件提供 Windows 普通 `cp314` wheel，但没有 Windows `cp314t` wheel，也没有该版本 sdist。普通 cp314 wheel 不能用于 free-threaded ABI。

本轮没有修改解释器目标、跳过 JAX 依赖或强制关闭 GIL。当前满足 3.14t 目标的 Windows 主机运行路径仍为 WSL2。原生 Windows 的 JAXlib 源码构建、普通 GIL Python 的替代运行属于另外的工作，本轮未实施。依据：[JAXlib 发布文件](https://pypi.org/project/jaxlib/0.11.1/#files)、[JAX 安装支持说明](https://docs.jax.dev/en/latest/installation.html)。

## 复验与产物

Linux/macOS 独立 checkout 示例：

```sh
uv build --python 3.14.7t
uv sync --locked --all-extras --no-editable
uv pip install --python .venv/bin/python --no-deps --reinstall dist/*.whl
.venv/bin/python -I scripts/check_runtime.py
.venv/bin/python -I scripts/check_distribution.py
.venv/bin/python -m pytest -q
```

实际测试时使用了独立目录作为 UV_PROJECT_ENVIRONMENT；上例采用默认 `.venv`。不要在 wheel 检查前用默认 `uv run` 自动重新同步成 editable 安装。

证据位于 `artifacts/platform-check/`：

- `windows/`：Python 信息、成功构建日志、依赖安装失败日志、sdist/wheel。
- `linux/`、`macos/`：构建/安装日志、runtime/distribution JSON、pytest 日志和 JUnit XML、sdist/wheel。
- `archive-check.json`：三个 wheel 的哈希、内容一致性及 sdist 生成文件排除检查。
- `jaxlib-wheel-availability.json`：检查时的 cp314 发布文件清单。

可以在任意普通 Python 环境运行 `python scripts/check_archives.py artifacts/platform-check` 复核包内容和哈希；该检查只使用标准库。
