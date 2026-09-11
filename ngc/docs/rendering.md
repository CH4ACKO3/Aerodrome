# 渲染数据接口：离线、在线与 FPS/TPS

`aerodrome.rendering` 提供渲染器无关的数据协议，不在物理积分中调用 GUI、文件或网络。当前已提供 JAX 姿态投影、JSONL 场景/轨迹文件、在线最新帧缓冲、离线插值以及 FPS/TPS 计数；没有内置 WebGL/Unity/Blender 渲染器或 WebSocket 服务。现已补上 [可替换后端与消息桥](renderer-backends.md)，统一生命周期、相机/关节请求和完成回执。

## 数据流与时间含义

```text
WorldState --纯 JAX project--> RenderSample（可在 scan 内记录）
                                    |
                            snapshot / device_get
                                    |
                        不可变 CPU Frame + Scene
                         /                     \
         TrajectoryWriter -> JSONL      LatestFrameStream -> renderer
                    |                                  |
          resample(FPS, speed)                   frame_presented()
                    |
            离线渲染器/视频编码器
```

| 指标 | 定义 |
|---|---|
| 物理 dt | 每个 tick 推进多少**仿真秒** |
| 名义 tick 频率 `1/dt` | 每个仿真秒有多少 tick，不是计算速度 |
| 实测 TPS | 完成的 physics tick 数 / 墙钟秒；step 含多个 tick 时按实际 tick 数计 |
| aggregate TPS | 批量下所有 world 完成的 tick 总数 / 墙钟秒 |
| 实测 FPS | 渲染端实际呈现帧数 / 墙钟秒；发布快照和生成离线帧不算呈现 |
| RTF | 推进的仿真秒 / 墙钟秒；与 `TPS * dt` 一致 |
| 目标 FPS | 显示/导出采样目标，与实际 FPS 和 TPS 独立 |

例如 dt=0.01 时名义频率为 100 tick/仿真秒；实际 TPS=2000 表示 RTF=20。画面仍可按 60 FPS 播放。8 个 world 批量推进时，每条时间线 TPS=2000、aggregate TPS=16000，不能据此宣称单 world RTF=160。

## 协议与坐标

`Scene` 是静态元数据，包含稳定实体 ID、可选 `asset_uri`、模型缩放、模型资产坐标→机体 FRD 的四元数，以及可选的局部原点经纬椭球高。资源 URI 仅作标识，接口不主动读取、下载或执行资源。

动态 `Frame` 包含 world ID、episode ID、仿真时间、真实 tick 或插值标记、源 tick 区间，以及按 Scene 实体顺序排列的姿态。每个姿态包含 NED 位置（米）和 Hamilton 标量在前 `[w,x,y,z]` 的机体 FRD→NED 四元数。JSON 使用 `schema_version=1` 并明确坐标和单位；读入器拒绝未知版本/坐标约定。

引擎采用 ENU、Y-up 或不同模型轴时，由渲染适配器使用已有 frames/rotations 工具转换；不能直接猜测分量交换。模型最终旋转为 `R_ned_body @ R_body_asset`。坐标原点和姿态方向必须保持一致，地理场景的 NED 位置须相对 Scene 的 `origin_lla`。

Frame 在 CPU 端复制为不可变元组，检查有限位置、有效四元数和 tick/time，四元数归一化不修改物理状态。动力学中间变量、控制器参数、气动表等不进入画面协议。额外仪表/曲线仍使用 telemetry，按 world/episode/tick/time 关联。

## 投影与批量 World

```python
from aerodrome.rendering import (
    Scene, RenderEntity, make_projection, rigid_body_pose, snapshot,
)

scene = Scene((RenderEntity("aircraft", asset_uri="models/aircraft.glb"),))
project = make_projection(.01, (
    lambda world_state: rigid_body_pose(world_state.entities[0]),
))
project = jax.jit(project)
sample = project(world_state)
frame = snapshot(scene, sample, world_id=10, episode_id=0)
```

Euler321 刚体使用 `rigid_body_pose(..., attitude="euler321")`。其他机体只需自定义 `WorldState -> Pose` 选择函数，不要求继承刚体状态结构。投影顺序必须与 Scene 一致。`make_projection` 使用当前 WorldState 的 tick/time，因此积分后的状态会标记为该 tick 的终点，不能将它与 World 的区间起点 record 混淆。

对 BatchState，先选择要看的 world，或 `jax.vmap(project)(batch_state.world)` 后取出某个 world 的 RenderSample；`snapshot` 明确拒绝直接输入批量/多时间轴。world ID 应从 BatchState.world_id 取，episode ID 从 episode_id 取，不用数组下标替代稳定身份。不同拓扑/坐标原点使用不同 Scene/流。

## 在线传输和节流

```python
from aerodrome.rendering import LatestFrameStream, RateLimiter, RateMeter

stream = LatestFrameStream()
publish_gate = RateLimiter(30.)
meter = RateMeter()  # 放在编译/warmup之后

# 仿真线程循环中：
state, records = compiled_step(state, inputs, parameters)
meter.simulation_completed(state, ticks=10, physics_dt_s=.01, world_count=1)
if publish_gate.ready():  # 先限流，再做投影传输
    stream.publish(snapshot(scene, project(state)))

# 独立渲染线程/事件循环中：
delivery = stream.take(timeout_s=0.)
if delivery is not None:
    renderer.update_scene(delivery.frame.to_dict())
# 渲染器在确实完成呈现后调用，重复呈现同一姿态也算实际画面帧：
meter.frame_presented()
```

以上 `renderer` 为适配器伪代码。`RateMeter` 以构造后的累计墙钟窗口统计，`snapshot()` 读数不会重置窗口；实时仪表需要滑动窗口时可在上层用计数差分。异步独立 world 各用一个 meter，避免把不同仿真时间线的秒数相加误作 RTF。

LatestFrameStream 为容量一的单逻辑消费者缓冲：新帧覆盖尚未读取的旧帧，计入 `dropped_unread`。它不会等待绘制完成，也不积压历史；锁仅保护元数据。可阻塞等待新帧、关闭唤醒，支持 episode 重置并拒绝旧 episode、倒退时间和换 world。多个独立消费者各建一个流。

`Delivery.sequence` 是递增传输序号；`published_wall_s` 是本进程 monotonic 时间，可计算帧龄，不能跨机器直接相减。WebSocket/IPC 适配器可发送 Scene JSON 和 `Frame.to_json()`，但网络服务、认证和远端时钟同步不在本次接口内。

`snapshot` 的 device_get 可能等待 GPU 并搬运数据，因此应在 UI 线程之外调用；最新帧缓冲不能消除传输成本。批量轨迹可先一次 device_get 整块投影结果，再发布块末尾帧。现有性能探针仍用于编译、计算、传输各阶段的耗时分析。

RateLimiter 只做非阻塞墙钟准入，错过周期后跳过，不补发一串过期帧；它不是硬实时调度器，不修改物理 dt，也不负责让仿真恰好以 RTF=1 运行。实时配速、暂停和慢放由应用层时间控制处理。

## 离线导出与重采样

```python
from aerodrome.rendering import TrajectoryWriter, read_trajectory, resample
with TrajectoryWriter("run.jsonl", scene) as writer:
    for frame in exported_snapshots:
        writer.append(frame)

scene, snapshots = read_trajectory("run.jsonl")
for display_frame in resample(snapshots, fps=60, playback_speed=1.):
    renderer.render_offline(display_frame.to_dict())
```

文件首行为 Scene，后续逐行 Frame；写入采用独占创建防止覆盖，支持持续 append/flush。仅保证提交的快照不丢，不等于自动记录全部物理 tick。它是可检查的 JSONL 协议；大规模数值记录仍宜用现有分块 NPZ 日志。读取器目前加载整段到宿主内存，长时任务应按 episode/时间分文件。

重采样按仿真时间作位置线性插值、四元数最短弧 SLERP，禁止外推和跨 world/episode 插值。显示帧之间的仿真间隔为 `playback_speed/fps`。插值帧的 tick 为 null，source_ticks 保留来源区间，不冒充物理结果。离线 60 FPS 表示输出时间轴，不代表编码器能每墙钟秒渲染 60 帧。

在线适配器也可保留最近两次 delivery 并调用 interpolate，但只能在已有时间区间内插值；没有新状态时保持最后画面或显示断流提示，不能自动外推。episode 切换时丢弃旧插值历史。

## 验证与示例

执行 `python examples/rendering_data.py`，每次生成一个新目录 `artifacts/rendering_data/<UTC>/`：

- `physics.jsonl`：100 Hz 物理采样，10 秒共 1001 个含初态快照。
- `replay_60fps.jsonl`：按仿真时间重采样，共 601 帧。
- `summary.json`：计时范围、TPS、RTF、发布/丢弃数量。

示例用真实 World/RK4/scan 计算，在线故意只在末尾消费，验证最新帧为 tick=1000；没有图形后端，实际 FPS 明确为零。不能把它的 TPS 当作复杂场景/GPU/实际渲染吞吐保证。

测试覆盖 JIT/vmap 投影、不可变快照、文件往返与版本检查、SLERP/反号四元数、固定 FPS 网格、重置边界、线程唤醒/丢帧/关闭，以及用确定性时钟验证 FPS、TPS、RTF 和节流。
