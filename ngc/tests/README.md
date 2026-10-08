# 测试

测试按实际行为分成两层：

- `modules/`：通过模块公开接口验证数值结果、物理性质、采样时序、批量与单独运行的一致性，以及实际的读写和失败恢复行为。
- `e2e/`：运行真实 CLI、Hydra 参数扫描、F-16 示例、教学实战项目和 HTTP 实验，检查最终状态、轨迹、指标及导出文件。HTTP 实验使用真实 JAX runner；不以假结果代替计算。

在 `ngc/` 中执行：

```sh
uv sync --locked --all-extras
uv run --locked --all-extras python -m pytest -q
uv run --locked --all-extras python -m pytest -q tests/modules
uv run --locked --all-extras python -m pytest -q tests/e2e
```

端到端测试在 pytest 临时目录中写入实验结果，不覆盖工作区产物。运行环境检查集中在 `e2e/test_runtime.py`，调用项目现有的 `check_runtime.py`。

新增测试应围绕一个完整模块行为或用户流程。数值测试使用解析解、独立实现、收敛性或物理不变量作对照，保留明确的误差容限。不要为每个参数校验、异常文案、私有字段或编译器内部布局单独建测试；正常重构不应要求同步修改这类断言。只有文件覆盖、外部调用失败、任务清理等实际影响结果或资源的失败流程，才保留相应的行为测试。
