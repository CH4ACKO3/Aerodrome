# 可替换渲染器模块

现有 Scene/Frame 数据协议之上新增 `RendererBackend`、`RendererSession`、`MessageBackend` 和 `HeadlessBackend`。本次实现引擎适配边界，不包含 Three.js/UE 程序、网络服务或图形资源。用户可以提供任意满足协议的适配器。

## 分层

```text
Python/JAX World -> RenderSample -> CPU Frame
                                      |
                     离线回放 / 在线 LatestFrameStream
                                      |
                              RendererSession
                           /                   \
               RendererBackend             MessageBackend
                 Python适配器                  RenderTransport
                                              浏览器/UE/独立进程
```

仿真和资源数据不依赖后端名称。依赖通过构造函数注入，不使用动态 import 字符串执行用户配置，也没有强制安装引擎 SDK。Scene 的 asset_uri 是资源逻辑标识，可由后端映射为 GLB、UE 原生资产或其他模型；同一语义模型在不同引擎的实际资产可不同。

## 生命周期与并发约定

| 方法 | 契约 |
|---|---|
| `open(scene, config)` | 加载静态场景，准备好后返回 Capabilities；失败须可 close 清理 |
| `submit(request)` | 非阻塞提交，成功后必须最终产生一个终态回执 |
| `poll()` | 非阻塞返回完成回执，无结果时返回空元组 |
| `close()` | 幂等，释放自有资源、取消或排空未完成工作 |

Session 在创建它的线程上调用后端，适合 GUI/引擎的主线程约束。物理线程只向已有 LatestFrameStream 发布不可变数据，不能跨线程直接调用 Session。独立进程/浏览器后端通过消息适配器隔离自己的线程调度。

Session 校验能力、实体拓扑、回执序号并维护 opening/ready/busy/failed/closed 状态。**最多一个在途请求**；忙时 submit 返回 None，表示未接收，调用者保留帧。实时消费端先确认 ready 再取最新帧；离线端在该帧完成前不前进。会话允许反向 seek/重复显示同一物理帧，回执按递增传输 sequence 区分。

`request_timeout_s` 默认 30 秒，poll 时检查；超时进入失败状态，应 close 后新建会话，不暗中重试，避免重复呈现。open 握手的超时由适配器负责。Session 不创建后台线程，也不会让本应非阻塞的错误后端自动变成非阻塞。持续轮询应接入应用事件循环，不要 busy-spin。

## 完成回执、FPS 与离线输出

- `presented`：引擎报告已实际呈现，Session 增加呈现帧计数。
- `completed`：离屏图像、文件或无头处理已完成，不虚增 FPS。可附 `artifact_uri` 指向后端产生的结果。
- `dropped`：实时后端有意放弃；离线模式视为错误。
- `failed`：后端错误，detail 描述原因，会话进入 failed。

接收/排队成功不是终态回执。重复、未知 sequence 的回执被拒绝。FPS、TPS、RTF 继续使用 RateMeter；可将仿真使用的 meter 显式传给 Session。Session 构造时默认创建的 meter 包含初始化等待时间，若需要纯运行窗口，应在 open 后换入应用预热之后创建的 meter。引擎自主重复刷新同一画面时，可通过应用明确调用 meter.frame_presented；不得与一次请求的 presented 回执重复计数。

## 相机与部件动画

`Capabilities` 声明 live/offline 模式和 camera/articulations 支持。请求未支持的能力会在提交前报错，不静默忽略。当前可选项包括：

- Camera：NED 位置、目标点、up 方向与垂直 FOV（弧度），校验退化视角。
- JointValue：实体 ID、节点名、角度（弧度）和归一化的节点 rest-local 转轴。每帧值为绝对偏角，不是累计增量；节点旋转应设为 `R_rest @ R_axis(angle)`。铰链原点和部件层级由模型资产定义。它不是整机 FRD 角速度。

有 joints 的请求应携带该帧所有需要控制的关节值；后端在新 world/episode 开始时重置节点到 rest pose，再应用新值，防止上个 episode 的舵角残留。缺失节点应返回 failed/detail。相机为空时使用后端默认相机，避免隐式继承未知相机状态。

Scene/Frame v1 姿态归档保持兼容。新增 RenderRequest 为单独的 protocol_version=1 消息封装；JointValue/Camera 尚未纳入原 TrajectoryWriter 的姿态轨迹归档，需要保留时可记录完整 RenderRequest JSON，或通过 telemetry 同步生成回放请求。地形流式加载、天气特效和动画片段控制应作为后续显式协议能力扩展。

## Three.js / UE 坐标边界

协议始终使用 NED、FRD、米与 Hamilton wxyz。`EngineCoordinates` 在适配器侧将位置与旋转矩阵转换到引擎基；支持正交的右手或左手基，检查两侧手系匹配，避免产生错误的反射姿态。

提供两种明确的约定预设（导入模型实际轴向仍由适配器处理）：

| 预设 | 世界方向 | 长度 |
|---|---|---|
| `THREE_Y_UP` | 东 +X、上 +Y、北 -Z；机体前 -Z、右 +X | 米 |
| `UNREAL_Z_UP_CM` | 北 +X、东 +Y、上 +Z；机体前 +X、右 +Y | 厘米 |

转换为 `position_engine = scale * A @ position_ned`、`rotation_engine = A @ R_nb @ B.T`，其中 A/B 为世界和机体系的换基矩阵。返回旋转矩阵，让引擎适配器按其 API 的矩阵布局/四元数存储方式转换，不直接复制或任意取反一个四元数分量。Scene 的 body_from_asset 修正也必须在资产层应用。

## 可执行基础后端

```python
from aerodrome.rendering import RendererSession, HeadlessBackend, RenderConfig

with RendererSession(HeadlessBackend(), scene, RenderConfig(mode="offline")) as renderer:
    for frame in replay_frames:
        sequence = renderer.submit(frame)
        receipts = renderer.poll()  # HeadlessBackend 立即完成；真实后端接入事件循环等待
```

HeadlessBackend 接受相机和部件值、仅保存最新请求并返回 completed，用于 CI/协议测试，绝不产生画面或 presented 回执。替换构造参数为自定义 RendererBackend 即可使用引擎。

执行 `python examples/renderer_backend.py`，重采样 1 秒轨迹、完成 61 个离线请求，输出 `artifacts/renderer_backend/scene.json`、`render_request.json`、`summary.json`，为后续前端提供真实序列化样例。

## 跨进程消息适配

`MessageBackend(RenderTransport)` 要求运输层提供 exchange/send/receive/close。exchange 用于有超时的初始化；send/receive 非阻塞且队列有界；receive 每次返回一个 JSON 字符串或 None。该接口可以实现 WebSocket、IPC 或 UE 桥，但本次没有启动这些服务。连接关闭时适配器负责通知/清理远端会话。

```json
{"protocol_version":1,"kind":"ready","capabilities":{"modes":["live","offline"],"camera":true,"articulations":true}}
```

初始 open 消息携带 Scene 和 RenderConfig；收到 ready 后发送 render 消息。终态回执示例：

```json
{"protocol_version":1,"kind":"receipt","sequence":1,"status":"completed","artifact_uri":"output://frame-000001.png"}
```

新连接使用新会话、重新加载场景，不能把旧连接的回执混入新序列。当前一个连接对应一个 Session；没有自动重连、认证、模型下载或远程文件访问。这些应由运输层/资源适配器按部署需求实现。

测试涵盖生命周期、能力拒绝、在途限额、超时、离线丢帧错误、重复回执、线程归属、消息往返、左右手系/厘米转换和相机/关节校验；不代表已在 Three.js 或 UE 中实机验证。
