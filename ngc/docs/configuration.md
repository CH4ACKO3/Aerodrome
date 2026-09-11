# 统一配置与实验运行

核心采用 **Pydantic 2.13.5 + PyYAML 6.0.3 + 标准库 argparse**，另提供可选 **Hydra 1.3.6** 实验入口。Hydra 的 Python 3.14 argparse 兼容处理位于项目适配器中，核心模型不依赖 Hydra。

Pydantic 负责 dataclass 类型、嵌套字段和组件参数校验；YAML 只表达数据；argparse 提供命令行。所有解析、资产读取、配平和模型构建在宿主侧完成，JIT/scan 内只传数组状态和参数。配置不会自动导入任意 Python 路径。

## 快速运行

以下命令在项目目录的 Linux / WSL 环境执行：

```sh
uv run --locked aerodrome --help
uv run --locked aerodrome --validate
uv run --locked aerodrome runtime.steps=100 runtime.chunk_steps=25
uv run --locked aerodrome --overlay renderer/headless.yaml runtime.steps=100
uv run --locked aerodrome --sweep 'seed=[1,2,3]' runtime.steps=100
uv run --locked --extra control aerodrome --overlay scenario/f16.yaml --overlay environment/earth.yaml runtime.steps=500
```

默认入口为包内 `configuration/conf/experiment.yaml`。`--config /path/experiment.yaml` 可换成自己的文件；`--overlay` 相对入口文件目录解析。`--validate` 会完成资产检查和模型构建（F-16 包含配平与控制器设计），但不推进仿真。

## 配置组合与校验

```yaml
includes:
  - environment.yaml
  - aircraft.yaml
schema_version: 1
name: course_demo
seed: 0
runtime:
  steps: 500
  chunk_steps: 100
  dtype: float64
  device: cpu
  jit: true
  trace: true
  output_dir: artifacts/config_runs
```

优先级为：includes 按顺序合并 → 本文件 → 命令行 overlays 按顺序合并 → 点路径覆盖。字典递归合并，列表整体替换；组件 `kind` 或 `version` 改变时整体替换该组件，避免继承旧模型参数。每个 include 相对包含它的文件解析。资产路径统一相对主配置文件目录；输出路径相对调用者工作目录。不会修改 cwd。

覆盖示例：`entities.0.model.parameters.mass_kg=1200`。覆盖只接受已存在的字段，包含已展开的默认字段；切换实体模型建议使用场景 overlay。未知字段、重复 YAML 键、循环 includes、错误类型、非有限数值、未知组件版本都会报错。版本使用字符串，例如 `version: "1"`。`steps=true`、`jit="false"` 不会被隐式转换。

`--sweep 'seed=[1,2]' --sweep 'runtime.steps=[10,20]'` 顺序运行四个独立实验，最多 256 组。这里是实验参数扫描，不是 `vmap` 的多 World 批处理；两种方式可以在后续批量 runner 适配器中结合。轻量入口不支持 Hydra 插值、resolver、launcher 或 `_target_` 语法；下面的 Hydra 入口提供原生实验组合功能。

## Hydra 实验入口

```sh
uv sync --locked --extra hydra --extra control
uv run --locked --extra hydra aerodrome-hydra --help
uv run --locked --extra hydra aerodrome-hydra --hydra-help
uv run --locked --extra hydra aerodrome-hydra action=validate
uv run --locked --extra hydra aerodrome-hydra --cfg job --resolve scenario=f16 environment=earth
uv run --locked --extra hydra aerodrome-hydra -m seed=0,1,2 renderer=headless runtime.steps=100
uv run --locked --extra hydra --extra control aerodrome-hydra scenario=f16 environment=earth runtime.steps=500
```

`aerodrome-hydra` 使用 Hydra 原生配置组、插值、命令行覆盖、multirun 和 launcher/sweeper 分发，随后将配置转成普通字典，进入同一个 Pydantic 校验、组件工厂和 runner。内置 scenario、environment、renderer 配置组由现有 YAML 预设注册到 ConfigStore，避免维护两份预设。组名和选项可用 `--help` 查看。

`aerodrome` 原有命令继续有效。两个入口有不同组合语法：轻量入口用 includes/--overlay/--sweep；Hydra 用 defaults、`scenario=f16` 和 `-m`。新增尚未出现在所选预设中的字段需使用 Hydra 的 `+path=value`；最终仍会由 Pydantic 拒绝未知字段和错误类型。没有添加自动 `_target_` 实例化，仍通过版本化组件工厂构建模型。

自定义 Hydra 主文件示例：

```yaml
defaults:
  - aerodrome_experiment
  - _self_
name: experiment_${seed}
seed: 7
asset_base_dir: .
runtime:
  steps: 100
```

保存为 `/path/configs/course.yaml`，然后执行：

```sh
uv run --locked --extra hydra aerodrome-hydra --config-dir /path/configs --config-name course
```

Hydra 入口的 `asset_base_dir` 默认是调用者原始工作目录；可显式指定配置文件目录或其他数据目录。相对 `runtime.output_dir` 也始终以原始工作目录为基准，因此 `hydra.job.chdir=true` 不会改变资产和仿真结果的位置。

Hydra 日志与 `.hydra` 配置保存在 `artifacts/hydra/<时间和随机后缀>/`，multirun 每组有独立子目录。仿真数组和统一快照仍写入 `runtime.output_dir` 下的独立实验目录；两边通过 `aerodrome-run.json` 和 `hydra-context.json` 相互关联。

Hydra multirun 默认顺序执行，未加轻量入口的 256 组限制；它不等于 JAX vmap。Launcher / Sweeper 插件需要另外安装并验证 Python 3.14t 兼容性。本次未声称支持已验证的集群或并行训练插件。Hydra 管理进程级配置、日志和可选 cwd，因此每个入口调用作为独立宿主任务运行，不要在同一进程的多个线程中并发调用 main。

### Python 3.14t 兼容处理

Hydra 1.3.6 用延迟对象提供 shell completion 帮助，Python 3.14 argparse 在添加参数时要求可格式化的字符串，导致原生入口报错。项目仅对自己的 parser 子类将该帮助对象转成字符串，保留其他参数的正常校验。

适配器复制 Hydra parser 工厂的函数全局命名空间，注入专用 parser 类；不修改 argparse 类、Hydra 模块全局或 site-packages。随后调用固定版本的 Hydra 内部分发入口，以保留原生实验管理流程。代价是提前生成补全帮助，以及依赖固定的私有 API；因此 extra 固定 `hydra-core==1.3.6`，升级时必须重新验证。上游讨论见 [Python 3.14 help 兼容问题](https://github.com/facebookresearch/hydra/issues/3121)。不提供 Hydra 的实验性 pickle rerun；数值记录不是完整对象检查点。

## 版本化模型、数据与扩展

统一入口包含 clock、runtime、entities、environment、renderer、assets。组件由 `category + kind + version` 定位，`parameters` 由工厂注册的 dataclass 校验。内置组件：

| 类别 | kind | 用途 |
|---|---|---|
| entity | rigid_body | 四元数 / Euler321 六自由度刚体、质量惯量、初值、恒定载荷 |
| entity | f16_longitudinal | 既有五状态纵向模型、配平和 DLQR，要求 float64 与正向 NED 重力 |
| environment | constant_gravity | NED 重力向量；vacuum 和 earth 为两个预设 |
| renderer | none / headless | 关闭渲染 / 无窗口后端；通过 every_steps 指定帧采样 |

F-16 仍是纵向教学模型，并未因此成为完整六自由度飞机。当前 F-16 配置工厂尚未提供渲染 Pose 投影，开启渲染会明确报错。

扩展示例：

```python
from dataclasses import dataclass
from aerodrome.configuration import builtin_registry, load_config, build_experiment

@dataclass
class GravityOptions:
    down_m_s2: float = 9.81

def factory(options, context):
    return context["vector"]([0., 0., options.down_m_s2], 3, "gravity")

registry = builtin_registry()
registry.register("environment", "course_gravity", "1", GravityOptions, factory)
config = load_config("experiment.yaml", registry=registry)
# 应用入口先设置所需 JAX 精度；build_experiment 不修改全局配置。
built = build_experiment(config, base_dir=".", registry=registry)
```

entity 工厂返回 `EntityBuild(assembly, initial, parameters, inputs, resources, pose)`，可封装已有图组合。每个实体的 resources 独立传递。JAX / MATLAB 混合图、设备 I/O、传感器、训练和评测 runner 尚未全部具备内置 YAML 适配器；本次提供它们共用的注册和验证入口，现有 Python API 保持可用。

```yaml
assets:
  aerodynamic_table:
    path: data/aero.bin
    sha256: "填写文件的64位SHA256"
    source: "数据来源和版本"
    license: "MIT"
    version: "1"
```

构建时校验 hash；工厂通过 `context["assets"]` 取得经过校验的 bytes。F-16 内置气动表沿用既有完整性检查，并将来源 manifest 写入运行记录。

## 实验记录与性能

每次运行创建带 UTC 时间和随机后缀的独立目录，记录展开后的 requested/resolved 配置、环境包版本、代码 SHA256、资产清单、设备信息、运行状态、最终状态及可选分块轨迹。NPZ 使用 `leaf_0` 等数组键，对应路径存于 `.paths.json`，可用 `np.load(..., allow_pickle=False)` 读取；它是数值记录，不是包含 Python 对象的恢复检查点。

`chunk_steps` 控制 scan 长度和记录内存，同一长度复用 JIT 函数；头次编译成本与稳定执行耗时不同。`trace=false` 避免生成轨迹。启用渲染会在帧采样边界结束 chunk，过高帧率会减少跨步融合的长度。当前 runner 采用固定实体输入，动态策略、实时外部 I/O 和向量化训练使用已有专用 API。

`device=cpu/gpu/auto` 显式选择执行设备。请求不存在的 GPU 会失败并留下失败状态，不会静默改成 CPU。当前验证环境使用 CPU JAX；CUDA wheel 仍需单独安装。
