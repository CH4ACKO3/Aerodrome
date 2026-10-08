# 教学工程：从组件到完整实验

先从[零号工程](project-zero.md)找到组件例子，再进入下列实战工程。每个实战工程都包含运行入口、传统方法、实际拟合的数据方法、独立验证场景的调优记录、留出评测和可回放结果。项目 README 是详细说明的唯一来源，教程站构建时自动导入；无需维护第二份教程。

当前五个工程是**低阶数学模型的首版**。尚未完成完整四旋翼、持续前向运动的固定翼绕障、覆盖协同或飞行器级故障恢复。数据方法包括回归、模仿和表格统计，不代表已经实现强化学习。

## 选择一个工程

| 工程 | 可运行的完整任务 | 比较方法 | 详细说明 |
|---|---|---|---|
| 01 约束运动规划 | 二维质点穿过圆障碍并停在终点 | A* 共用路径下的解析速度参考 / RBF 模仿参考 | [规划与控制](../projects/obstacle_navigation/README.md) |
| 02 动态跟踪与离散决策 | 外生信号跟踪；独立三动作重复博弈 | 标称 / 辨识模型；均匀 / 历史频率 / 学习转移表 | [跟踪与决策](../projects/dynamic_decision/README.md) |
| 03 变人数协同 | 成员加入退出后恢复移动队形 | 独立反馈 / 局部一致性 / 共享线性模仿策略 | [变人数协同](../projects/variable_team/README.md) |
| 04 任务调度 | 处理节点与作业匹配、动态到达与完成 | 贪心 / 标称代价分配 / 学习处理时间分配 | [任务调度](../projects/task_scheduling/README.md) |
| 05 效能退化适应 | 一维执行效能下降后的跟踪恢复 | 固定反馈 / 理想补偿 / 在线辨识 / 近邻估计 | [适应控制](../projects/adaptive_control/README.md) |

## 在本地运行

在 `ngc/` 中安装锁定依赖。导航工程绘图使用已有 `control` extra 带来的 Matplotlib，其他工程只使用核心数值依赖。

```sh
uv sync --locked --all-extras
uv run --locked --all-extras python projects/obstacle_navigation/run.py --output artifacts/projects/01
uv run --locked --all-extras python projects/dynamic_decision/run.py --output artifacts/projects/02
uv run --locked --all-extras python projects/variable_team/run.py --output artifacts/projects/03
uv run --locked --all-extras python projects/task_scheduling/run.py --output artifacts/projects/04
uv run --locked --all-extras python projects/adaptive_control/run.py --output artifacts/projects/05
```

这些是 Python CLI，目前没有接成网页运行按钮。01/02/04 可指定 `--seed`，03/05 使用源码和导出配置中列出的固定训练/验证/测试种子；各自 README 说明默认参数和数组字段。重复使用相同输出目录会覆盖旧结果，比较实验时使用不同目录。

01/02/04 输出 `summary.json`，03/05 输出 `metrics.json`；都保留训练数据、拟合模型与评测轨迹。01 另有直接可读的 JSON 轨迹和静态图，其他工程主要使用 NPZ。接网页时读取这些原始结果，避免用另一套网页公式生成看似相同的响应。

## 怎么阅读和调优

先读模型、观测和动作约束，再读传统方法，随后追踪训练数据从哪里来、标签是什么、参数怎样拟合。最后看验证集如何选参数，以及留出案例中的失败或退化。

学习法不保证更好：导航的模仿只替换局部参考，未替换 A*；离散决策的冻结转移模型遇到规律反转会退化；调度时间预测改善也可能不改变静态分配；本批故障例中在线辨识优于学习估计。这些结果应保留用于教学。

代码尽量使用普通函数和薄的 CLI，没有项目基类或统一策略继承层。共同的恒加速度运动由 `aerodrome.models.point_mass` 负责；带非线性阻力的工程 02 在 RK4 各阶段重新计算加速度。短实验不要求套入 World，多速率或异构组件组合时再复用 World。

## 教材章节与零号工程

章节讲原理，示例链接到[零号工程固定小节](project-zero.md)；实战工程反向链接自己需要的组件。以编程附录为例：

```md
[运行示例：三种代码写法](/Aerodrome/reference/project-zero/#coding-styles)
```

三种写法的同一速度闭环见[代码编写教学](code-styles.md)。完整后续路线保留在 `ngc/design/teaching-projects.md`，不能把规划表中的功能当成已经实现。
