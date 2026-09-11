# Python/JAX 重构进度存档

本目录保存 2026-09-11 完成的 Aerodrome Python/JAX 架构原型；仓库根目录保留本科毕业设计的旧版 C++/Python 实现。

`ngc/` 是独立 Python 项目，安装和测试请先进入该目录。当前运行目标为 CPython 3.14.7t，依赖锁定于 uv.lock；详见 README.md 与 docs/platform-builds.md。Linux x86_64 和 macOS ARM64 实测各 211 项测试通过，Windows 原生运行受到 JAXlib cp314t wheel 缺失限制。

本次迁入不包含 artifacts、虚拟环境、缓存或构建产物。历史运行路径记录属于原型工作区的验证证据，不是本仓库内可用的下载链接。工作区专用 scripts/run-wsl.ps1 依赖原来的 work/ 布局；独立 checkout 使用以下命令：

```sh
cd ngc
uv sync --locked --all-extras
uv run --locked --all-extras python scripts/check_runtime.py
uv run --locked --all-extras python -m pytest -q
```

后续文档站在独立 docs 分支开发，核心实现的此版本保存在 main。
