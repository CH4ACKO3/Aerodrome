# 日志、传感器与性能探针

可运行示例：`uv run --locked python examples/instrumented_rollout.py`。Windows 也可运行 `./scripts/run-wsl.ps1 examples`。示例生成批量传感器日志、性能统计 JSON 与 Chrome/Perfetto trace JSON，保存在 `artifacts/instrumented_rollout/run-*/`。

## 日志：数值投影与文件写入分开

`telemetry.Recorder` 是纯函数，按命名 Channel 选择需要的量。可以直接作为 `BatchedWorld.rollout(..., record=recorder)` 的记录投影，不增加模型端口或把文件 I/O 放进 JIT。

```python
recorder = Recorder((
    Channel("pitch", lambda world, trace: world.entities[0].truth.pitch_rad, "rad"),
    Channel("tick", lambda world, trace: world.tick, "tick"),
))
run = jax.jit(lambda s, u, p: batch.rollout_constant(
    s, u, p, steps=128, record=recorder))
writer = ChunkWriter("artifacts/my-run", recorder, axes=("time", "world"),
                     metadata={"seed": 42, "world_ids": [100, 101]})
state, records = run(state, inputs, parameters)
writer.write(0, records)
```

每个 chunk 保存压缩 NPZ 数组及 JSON 单位、说明、轴、shape、dtype 和实验元数据。写入在宿主侧显式执行 `device_get`；当前是同步单写入者，不包含后台队列。通道必须共享声明的前导轴，连续 chunk 只允许第一维长度变化。已有文件拒绝覆盖，`np.load(..., allow_pickle=False)` 即可读取；读者只接受带可解析 JSON sidecar 的 NPZ。失败写入可能留下不完整文件，当前不提供崩溃恢复或事务数据库保证。

`Channel.select` 的参数由调用方决定，也可以对整个 `Transition` 使用 `recorder(transitions)`。训练日志应显式选择 `world_id`、`episode_id`、`terminated`、`truncated` 和 terminal `next_observation`；不要仅凭时间戳猜 episode 边界。记录真值用于评测不会自动将真值暴露给策略。

## 传感器：可连线的 Module

`models.sampled_sensor.SampledSensor` 将已有物理信号端口转换为测量端口，保持物理单位、shape、frame 和 quantity。它实现：

`measurement = clip(quantize(truth + bias + drift + noise_std * normal_noise))`

其中 `drift` 按采样间隔进行随机游走，增量标准差为 `drift_std_per_sqrt_s * sqrt(every * physics_dt_s)`。每次计划采样都会推进漂移，即使该数据包丢失；第一次在 tick=0 采样时也应用一次漂移增量。多通道噪声与漂移独立，丢包以整个数据包为单位。

```python
sensor = SampledSensor("gyro", angular_rate_port, every=4, delay_ticks=2)
module = sensor.module()
initial_value = sensor.initialize(key)
params = SensorParameters(noise_std, bias, drift_std_per_sqrt_s,
                          dropout_probability, resolution, lower, upper)
params.validate(angular_rate_port.shape)  # 宿主配置阶段调用
```

通过 `Wire` 将状态映射模块的输出连接到 `gyro.truth`；需要坐标变换或观测函数 `h(x)` 时先用独立模块计算，传感器不隐式读取整个机体状态。`GraphAssembly` 的初始模块值可由工厂生成；批量自动重置场景使用 `BatchedWorld.initial_sampler` 根据新 episode 的 key 重建传感器初态，不能把同一个固定随机键复制给所有 World。

输出语义：

| 输出 | 含义 |
|---|---|
| `value` | 最近成功送达的测量值；未收到时为零占位 |
| `sample_tick` | 该测量的采样时间；未收到时为 -1 |
| `valid` | 是否至少收到过一个有效测量，之后保持 True |
| `fresh` | 当前物理 tick 是否有新测量送达，只维持一个 tick |

`every` 控制内部采样，`delay_ticks` 控制固定传输延迟。底层 Module 始终每 tick 执行，以处理到达队列并清除 fresh。禁止把其 `Module.every` 再改为采样周期，否则会跳过送达时刻。滤波器只在 `fresh & valid` 时校正，不能把保持值当作新测量重复融合。新旧程度可以由 `current_tick - sample_tick` 判断；`valid` 本身不代表数据足够新。

采样键按传感器 ID 和 tick 派生，按 chunk 续算不会改变随机序列。当前使用 threefry2x32，支持 scalar/vector 浮点信号与 JIT/vmap；不模拟相关噪声、温度漂移、随机时延或真实硬件协议。分辨率零表示不量化，正值使用 `jnp.round` 的舍入规则。旧 `measure_pitch` 接口继续保留，已有教学回归不变。

示例增加的是一维速度模型上的诊断传感器分支，原控制器仍使用理想观测。正式替换导航输入时应连接测量及 freshness，并对延迟观测选择适当的滤波更新方法。

## 性能：区域、模块与整段 rollout

`PerformanceProbe.call` 对输入先等待就绪，开始计时后调用函数，并等待返回数组完成。耗时是同步后的宿主墙钟时间，含调用、运行时调度和输出等待；不含输入产生时间，不是独占设备 kernel 时间。异常也会记录 status，并原样抛出。探针可由多个执行线程共享。

```python
probe = PerformanceProbe()
program.run(state, inputs, parameters, probe=probe, phase="cold")
# 只有可安全重复执行的纯 JAX 模型才适合用相同输入预热。
program.run(state, inputs, parameters, probe=probe, phase="warm")
probe.write("artifacts/performance.json")
```

默认融合计划记录每个区域的实际耗时和其覆盖的模块/tick。JAX 融合后没有可靠的逐源代码模块时间边界，不能把区域时间平均分配给模块。

需要逐模块诊断时，另建 `CompiledModuleGraph(..., fuse=False)`；每个更新事件独立计时，报告按模块名汇总次数、总和、平均、最小和最大值。未到调度周期的模块不产生更新事件。关闭融合和额外同步会改变吞吐，不能将该诊断结果当作正常融合运行中的耗时分解。并发区间可能重叠，各模块耗时之和不等于总墙钟时间。

第一次 JIT 调用可能包含编译，区域事件明确记录 `compilation: included on cache miss`。`phase` 是调用方标签，不自动证明已预热。对纯 batch/native 路径，示例单独计时 `lower(...).compile()`，然后执行预热和正式测量。宿主/MATLAB 模块可能有外部状态，框架不会自动重复调用来预热，必须由适配器拥有者负责安全重置。

原生图调用添加了 `jax.named_scope(module_id)` 以辅助后续 JAX profiler 定位；这只是名称注解。需要 GPU kernel/设备时间线时再使用设备 profiler，目前没有实测 GPU 计时。当前探针本身不做模型文件 I/O、不进 JIT，也不改变传感器随机状态。
