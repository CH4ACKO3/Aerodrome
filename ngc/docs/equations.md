# 公式—实现—验证索引

约定：theta 为俯仰角 [rad]；q 为俯仰角速度 [rad/s]。这是二状态小模型，不涉及六自由度坐标转换。正舵偏按该案例定义为产生正俯仰角加速度。

| 公式 ID | 内容 | 实现 | 验证 |
|---|---|---|---|
| pitch.linear | theta_dot=q; q_dot=-a theta-b q+c delta | models/pitch.py:rhs | 矩阵指数解析解与 RK4 收敛阶 |
| integration.rk4 | 四阶段完整 RHS 重算 | models/pitch.py:advance | 步长减半误差约缩小 16 倍 |
| sensor.pitch | y=theta+sigma epsilon | models/sensors.py | 控制调度改变不改变噪声序列；独立种子 |
| kf.discretize | F=exp(A dt), G=integral exp(A t)B dt | navigation/kalman.py:discretize | 双积分器的 F、G、Q 闭式解 |
| kf.predict | x-=F x+G u; P-=F P F^T+Q | navigation/kalman.py:predict | 多速率无新测量时保留 prior |
| kf.correct | K=P H^T/(H P H^T+R), x+=x-+K(y-H x-) | navigation/kalman.py:correct | 协方差对称、非负；编译/批量一致 |
| guidance.slew | r_new=r+clip(goal-r, +/-rate*dt_g) | guidance/pitch.py:update | 更新时间和保持行为 |
| control.pitch_pid | delta=clip(kp(r-theta_hat)+ki I-kd q_hat) | control/pid.py:update | 饱和时不继续向外积分；舵限 |

## 连续模型与导航模型

物理模型用 RK4；导航的离散模型使用零阶保持的矩阵指数。两者不是同一个积分实现，所以可以检测物理积分误差。模型系数仅为展示代码结构选择，不来自某架飞机的辨识结果。

## 协方差

测量 H=[1,0]。后验采用 Joseph 形式：

```text
P+ = (I-KH) P- (I-KH)^T + K R K^T
```

连续过程噪声强度 W=diag(0,spectral_density)，由 Van Loan 方法计算：

```text
Qd = integral_0^dt exp(A*t) W exp(A^T*t) dt
```

其意义是估计器对未建模角加速度的容许量；默认真实模型没有过程噪声。因此这里不把后验协方差视作已通过统计校准的置信保证。覆盖率与创新统计应在未来匹配噪声的实验中验证。

## 教学限制

本次测试验证代码与上述数值契约，不代表飞机飞行包线验证、PID 最优性、滤波一致性或 GPU 性能验证。没有实现气动参数、执行器滞后、传感器偏置与真正航路制导。
