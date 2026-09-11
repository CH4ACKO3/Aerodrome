# 键盘、鼠标、摄像头与外部 I/O

新增 `adapters.device_io`、`adapters.input_devices` 和 `runners.world_io`。设备/UI 线程采集，固定形状的数值快照进入仿真；计算结果由独立输出通道交给设备/界面线程。导入这些模块不会监听键盘、打开窗口或启动摄像头。

## 两条接入路径

| 路径 | 接法 | 时间与编译性质 |
|---|---|---|
| World / 原生 JAX 图 | WorldIO.sample_inputs → 显式输入 → GraphAssembly/World.step | 一次 step 的输入保持；JIT/vmap/scan 保留 |
| 混合计算图 | InputChannel.module(host) → JAX 模块 → OutputChannel.module(host) | 现有 CompiledModuleGraph 在 host 边界拆分；采样频率由 every 决定 |

Host source 模块只读内存快照，不执行摄像头 read 或操作 GUI。输出 sink 只入队，不在计算图工作线程上绘制画面或控制设备。带 host 模块的图不能放入纯 JAX World，使用混合编译器；原生 World 通过 WorldIO 在宿主边界连接。

## 通道语义

`InputChannel(ports)` 使用既有 Port 定义字段名称、形状、dtype、单位和坐标系。publish 必须严格匹配，拒绝非有限值；复制为独立存储。容量为一，每个读者通过自己的 previous_sequence 判断新鲜度，不会抢走另一个读者的数据。

每个读取快照额外包含：

- `io_sequence`：int32 单调样本序号，耗尽时报错，重新创建通道。
- `io_valid`：bool，数据已连接且未超龄；关闭/断连时为 False。
- `io_fresh`：本次读取是否观察到新且有效的序号。
- `io_age_s`：float32，在本地 monotonic 接收时钟中的年龄，不是仿真时间，也不代表摄像头实际曝光时间。

超龄或断连仍保留最后的数值，并标记无效。**控制算法必须显式处理 valid**，例如输入归零、保持安全参考或暂停，不应直接使用旧控制量。设备时钟与仿真时间没有隐式映射；离线高速仿真可能连续读到同一物理输入，这是预期行为。确定性回放应记录实际采样输入，并绕过实时设备。

## 键盘和鼠标

```python
from aerodrome.adapters.input_devices import KeyboardMouse
hub = KeyboardMouse(("pitch_up", "pitch_down"))

# 浏览器、UE 或本地 UI 的事件处理程序调用：
hub.key("pitch_up", True)
hub.key("pitch_up", False)
hub.motion(position=(120., 80.), delta=(2., -1.))
hub.button(0, True)  # 0左、1中、2右
hub.wheel((0., 1.))
hub.heartbeat()     # UI每次poll都调用，没有事件时也表明输入源仍连接
sample = hub.channel.read()
```

字段包含 keys/buttons 的保持状态、presses/releases 和 button_presses/button_releases 的累计计数、像素 position/motion_total、归一化约定下的 wheel_total、focused。累计计数保存短按数量，按键自动重复不会重复增加按下计数；它不是保序事件日志，不能恢复多个事件的先后顺序。

鼠标坐标/移动为 UI 像素，向右/向下为正；浏览器适配器需明确 CSS 像素与设备像素，滚轮需先归一化为约定单位。计数差和累计位移差由控制模块按序号计算，避免一个保持快照重复积分。每次 episode 初始化，应使用当前计数作为基线，不能把历史累计计数误认为本回合新事件。

`focus(False)` 清除按住状态、补记释放，防止失焦卡键。PygameInput 接收应用已经获取的 event 列表，不创建窗口、不抢走事件队列，也不监听系统全局键盘：

```python
from aerodrome.adapters.input_devices import PygameInput
adapter = PygameInput(hub, {pygame.K_w: "pitch_up", pygame.K_s: "pitch_down"})
# 在已有 pygame 的主线程事件循环：
events = pygame.event.get()
adapter.handle(events)
# 应用仍可以处理同一组 events，包括窗口退出等事件。
```

Pygame 为懒加载可选依赖。参照 [Pygame 事件文档](https://www.pygame.org/docs/ref/event.html)；此适配器处理按键、鼠标、滚轮及窗口焦点事件。

## 摄像头

```python
from aerodrome.adapters.input_devices import OpenCVCamera
camera = OpenCVCamera(width=320, height=240)
# 以下生命周期全部放到同一个采集线程或进程，明确调用才会打开设备：
camera.open(device=0)
try:
    while running:
        if not camera.read():
            break
finally:
    camera.close()
```

输出 `rgb: uint8[H,W,3]`，先 BGR→RGB，再按声明尺寸缩放；这保证 JAX 固定形状，缩放不是相机几何标定。设备失败、视频结束或异常会使通道无效。read 可能在设备驱动内阻塞，不能放进计算图工作线程；要求强制取消时应隔离到可管理的采集进程，而非从另一个线程强行 release。

OpenCV 仅在 open 时导入。[OpenCV 颜色说明](https://opencv.org/color-spaces-in-opencv/)解释了默认 BGR 顺序。本项目的 Python 3.14t/WSL 环境未安装或验证 Pygame/OpenCV 的硬件依赖；不要假设普通 CPython 的 wheel 可直接用于 free-threaded 构建。可在能访问设备的宿主进程运行采集，再由用户的 IPC 适配器向通道 publish 固定格式数组。没有自动启用 Windows 摄像头透传或权限设置。

## World 接入

```python
from aerodrome.runners.world_io import WorldIO
runner = WorldIO(
    world, channels={"controls": hub.channel},
    selectors={"aircraft": lambda samples: make_aircraft_inputs(samples["controls"])},
    max_age_s=.5,
)
state, record, actual_inputs = runner.step(state, parameters)
```

selectors 必须覆盖 World 的所有实体，并产生实体自己的输入 PyTree。make_aircraft_inputs 是用户的按键/鼠标→控制量映射，应显式检查 valid，避免库猜测按键控制的物理含义。WorldIO 每个 step 采样一次，返回 actual_inputs 用于记录/重放；多速率细节仍由 World 管理。需要更高采样频率时缩短 step。

图内纯输入模块使用 `input_module(name, ports)`，通过 InputBinding 接入图的外部端口。普通端口的初态为 ModuleValue((), outputs)；包含完整 io 元数据时初态可用 channel.initial_value()，模块在状态中保存上次序号，在后续模块执行时将同序号的 io_fresh 置 False。every>1 时，未执行 tick 的全部输出按图规则保持；下游仍需按序号/累计计数去重，不将保持的 fresh 当成新事件。绕过该模块直接使用 WorldIO 数据时，控制模块应自行保存上次序号/累计计数，不依赖宿主 sample 的 fresh 每个 tick 都只触发一次。

批量 World 通过既有 BatchedWorld.input_axes 显式决定共享输入还是每个 world 独立输入。WorldIO 当前是普通 World 的单宿主循环包装；不要将同一真实设备不加说明地复制成多份独立随机观测。训练回放可以直接将采样序列送进原生图/批量 rollout。

## 混合计算图和输出

```python
source = hub.channel.module("keyboard_mouse", every=2, max_age_s=.5)
sink = output_channel.module("ui_output", every=4)
# 将 source 的端口通过 Wire 接入控制模块，再接 sink。
# 用 CompiledModuleGraph 执行；参数中给 source/sink 传 {}。
```

输入 source 初始值使用 channel.initial_value()。输出 sink 初始值为 `ModuleValue((), {"queued": np.asarray(False)})`。queued 只代表成功入队，不表示外部设备已执行或画面已呈现。

`OutputChannel(ports)` 是最新输出命令缓冲，取出 `(tick, time_s, values)`；未取出的旧命令会被覆盖并计数。UI/设备线程自行 take 并调用其本地 API。适合最新姿态、UI 仪表、持续目标值；不适合必须执行每条的离散动作，应使用有顺序和确认的专用传输。渲染呈现确认继续使用已有 RendererSession，而不是把 queued 算为 FPS。

WorldIO 的 outputs 参数可传 `(channel, selector)` 元组列表，selector 接收 following_state/record，结果在 step 结束后投递。两个方向的通道均由应用管理关闭，WorldIO 不擅自关闭共享设备。摄像头、键鼠事件流、输出缓冲各自有独立生命周期。

## 验证范围

测试涵盖宿主新鲜度/超时、累计短按与失焦、图内 host 源/输出 sink、World 接入和输入重放、原生图保持输入的新鲜度去重、模拟 Pygame 事件和模拟摄像头 BGR→RGB/失败/释放。完整回归未打开真实键盘监听或摄像头；实际 OS 设备、窗口和网络传输仍需在目标宿主集成验证。
