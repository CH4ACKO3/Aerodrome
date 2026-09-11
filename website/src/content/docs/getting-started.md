---
title: 在线阅读，本地运行
---

在线版与本地版使用同一份静态站点。页面根据实例连接状态显示可用操作，手册始终可以阅读。

## 在线版

直接打开 GitHub Pages 即可。实验页面展示标注参数的预计算参考轨迹，可下载配置；修改参数不会伪造新的“仿真结果”。无需 Python，也无需登录。

## 本地版

先在仓库根目录构建同一套站点（Node.js 24）：

```sh
cd website
npm ci
npm run build
cd ../ngc
uv sync --locked --all-extras
uv run --locked --all-extras aerodrome-teach --site ../website/dist
```

打开终端显示的本机地址。服务默认监听 `127.0.0.1:8765`，同时提供页面与实验 API；浏览器会获得仅限该本地会话的凭据。不要直接双击 HTML 文件，也不要把在线站点连向任意本机端口。

Windows 当前通过 WSL2 运行 Python 3.14t / JAX；Node.js 构建可以在 Windows 原生完成。Linux x86_64、macOS ARM64 的运行验证见[三平台报告](/Aerodrome/reference/platform-builds/)。

## 连接状态

- **在线手册**：没有连接本地实例，仍可读取参考数据。
- **本地实例就绪**：API、核心版本和实验能力均匹配，可以提交实验。
- **版本不匹配**：页面继续可读，实际运行按钮禁用；重新构建对应版本手册，或切换相同版本的 Python 环境。
- **连接中断 / 运行失败**：页面给出具体提示，可重新连接或重新提交，不自动重复运行。

本地实验使用固定白名单接口，目前开放刚体速度控制。不能上传并执行任意 Python 代码。任务在本机线程池内运行，页面轮询进度与结果；实时 WebSocket 遥测和其他实验会作为后续适配器扩展。
