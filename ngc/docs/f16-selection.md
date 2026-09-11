# F-16 baseline 来源选型

审阅日期：2026-09-11。结论：推荐采用 **ISRL F16-Model-Matlab 的气动与机体实现作为首个 baseline 的参考来源**，由 Aerodrome 逐模块移植、验证和组装。该推荐不是宣称现有 JAX 原型已经接入 F-16，也不把外部仓库称为经过本项目认证的全包线飞机。

后续进展：已实现这份数据的 beta=0 对称纵向子模型，并完成理想状态反馈下的配平与扰动恢复实验，见 [F-16 纵向平飞](f16-level-flight.md)。下文保留选型时的范围说明；完整六自由度机体、真实发动机/舵机和 MATLAB/Julia 对照仍未完成。

## 推荐来源

- 仓库：[isrlab/F16-Model-Matlab](https://github.com/isrlab/F16-Model-Matlab)。ISRL 属于 Texas A&M University。
- 本次锁定提交：`d019742f2fe7f5f25c521bc7e846519b28b759b3`。
- 仓库许可证：MIT，版权声明为 2023 Raktim Bhattacharya。移植时保留声明，并逐资产保存来源信息。
- 作者说明其气动来源为 NASA TP-1538，模型基于 Stevens & Lewis 的 Aircraft Control and Simulation。
- 原始文献：[NASA TP-1538 / 19800005879](https://ntrs.nasa.gov/citations/19800005879)。研究基于 F-16 风洞数据的战斗机构型，不应描述成任意现代 F-16 批次的数字孪生。

## 已检查的具体资产

| 文件 | 用途 | 处理方式 |
|---|---|---|
| F16AeroData.h5 | 可独立读取的气动数据 | 作为原始资产锁定；后续转换为带轴语义的数值数组 |
| preprocess_F16_AeroData.m | 数据表布局和插值构造 | 作为轴顺序、reshape 和插值语义的参考 |
| F16AeroFM.m | 气动力、力矩和修正项 | 对照公式后实现 JAX 纯函数 |
| load_F16_params.m | 几何、质量、惯量、重心 | 单独参数包；核对单位与惯量积符号 |
| F16.slx / F16_2023a.mdl | Simulink 机体实现 | 后续作为 MATLAB 侧对照，不要求基础安装依赖它 |
| trim_and_linearize.m / TrimF16.m | 配平和线性化 | 提取参考工况；对应 MATLAB 工具箱依赖另行记录 |
| test_with_julia.m / F16_Julia_Dump.mat | 跨语言对照材料 | 用于理解验证流程，不把脚本存在等同测试通过 |

气动 HDF5 SHA-256：`b9ac8d21cfb749c0e9897766d5f7b3cdcfdafc86ef8e948c064aafdc2e453f07`。

### 必须保留的数值语义

1. 预处理使用 MATLAB `griddedInterpolant(..., 'linear', 'none')`。JAX 移植要复现多线性插值及域外无外推语义；不能静默改为边界截断/外推。
2. MATLAB reshape 与 Python 数组布局不同，必须用节点值和非节点值对照确认轴顺序，不能仅以 shape 相同判断一致。
3. 气动函数将输入角度从弧度转为度后查表。角度单位应记录在数据 manifest 与接口适配中。
4. 原实现存在重心相关力矩修正。组装时确认力矩参考点，避免在统一力矩汇总层再次平移同一项。
5. `F16AeroFM.m` 明确将 `delta_Cm_ds` 置零，注明忽略 deep-stall effects。该限制进入模型卡与评估包线。
6. Simulink 机体的公开端口包含 Thrust。选中的是机体/气动来源，不表示已经获得与之匹配、已验证的完整发动机动态。发动机与舵机必须单独作为组件选择和验证。
7. MATLAB/Julia 对照材料有英制与 SI 混合的注释和转换，移植以实际公式、端口和数据为准，不机械复制注释。

源码：[气动函数](https://github.com/isrlab/F16-Model-Matlab/blob/d019742f2fe7f5f25c521bc7e846519b28b759b3/F16AeroFM.m)、[插值预处理](https://github.com/isrlab/F16-Model-Matlab/blob/d019742f2fe7f5f25c521bc7e846519b28b759b3/preprocess_F16_AeroData.m)、[参数](https://github.com/isrlab/F16-Model-Matlab/blob/d019742f2fe7f5f25c521bc7e846519b28b759b3/load_F16_params.m)。

## 为什么不直接选其他仓库

- [F16Model.jl](https://github.com/isrlab/F16Model.jl)：同实验室、MIT，有配平与自动微分线性化，适合第二实现对照；README 明确机体模型不含执行机构动态。两实现有共同来源，其相符不是完全独立的物理验证。
- [F-16_Bristol](https://github.com/duchn7/F-16_Bristol)：非常贴近本科教学，有 4/8/12 阶模型、MATLAB/Simulink/Julia 版本和相关教学论文。审阅提交 `061019ed152583aa3174e30540a9ff8651dad63d` 未找到明确 LICENSE 文件，暂作为教学结构参考，不直接搬运代码。其[大学资料页](https://research-information.bris.ac.uk/en/datasets/f-16bristol/)说明了研究和教学背景。
- [AeroBenchVVPython](https://github.com/stanleybak/AeroBenchVVPython)：可参考 F-16 验证案例与发动机函数，但仓库标记 GPL-3.0；不把它的发动机代码无说明地合入 MIT 来源实现。
- NeuralPlane/AeroPlanax：可以参考 GPU 执行方式；前者的气动代理和后者的插值边界处理不应自动成为本 baseline 的数值标准。

## 下一步的模型引入门槛

先完成气动数据导入与语义验证，再实现气动输出，再接六自由度 RHS。逐级验证：数据轴与节点 → 插值点 → 力/力矩 → 同状态同输入导数 → 配平残差 → 小扰动轨迹 → 局部线性化。发动机、执行器选定后，再执行完整闭环验证。

选型时尚未运行 MATLAB/Julia或验证轨迹；现在仅验证了上述纵向实验，仍无包线或六自由度认证。对外发布的 baseline 名称建议体现家族和版本，例如 `f16.nasa1538.isrl-airframe.v1`；完整组合另有 assembly ID，不能只写 `F16`。
