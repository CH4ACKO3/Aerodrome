"""数学基础的概念配图：用本站算例解释几何、概率、决策与递推。

在 website/ 运行 python scripts/plot-math-foundations.py --font /path/to/font.ttf。
所有曲线为教学设定下的解析结果，不是实测数据；不需要随机采样。
SVG 用于网页，PNG 用于检查文字、遮挡和裁切。布局参考 Murphy 的插图用途，
数值、构图和绘图代码均为本站独立编写；具体对应关系见 DESIGN.md。
"""
import argparse
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font', required=True, type=Path)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'public/figures/math-tools')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
plt.style.use(Path(__file__).with_name('teaching.mplstyle'))
font = FontProperties(fname=args.font)
blue, orange, gray, purple = plt.rcParams['axes.prop_cycle'].by_key()['color']
root = Path(__file__).resolve().parents[2]
examples = runpy.run_path(str(root / 'ngc/examples/math_tools.py'))
dp = runpy.run_path(str(root / 'ngc/examples/finite_horizon_dp.py'))


def save(fig, name):
    """只共用输出与字体设置；每幅图的数学和布局保留在相邻代码中。"""
    for i, ax in enumerate(fig.axes):
        if len(fig.axes) > 1:
            ax.set_title(f'({chr(97 + i)}) {ax.get_title()}')
        labels = [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.title,
                  ax.xaxis.label, ax.yaxis.label, *ax.texts]
        if ax.get_legend():
            labels += ax.get_legend().get_texts()
        for label in labels:
            size = max(12, label.get_fontsize())
            label.set_fontproperties(font)
            label.set_fontsize(size)
    fig.savefig(args.output / f'{name}.svg', bbox_inches='tight')
    fig.savefig(args.output / f'{name}.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(name)


# 1.1：用两个不正交的基合成一个位移。虚线保留首尾相接的加法过程。
fig, ax = plt.subplots(figsize=(7.5, 4.6), layout='constrained')
a1, a2 = np.array([1., 0.]), np.array([1., 1.])
for start, end, color in [(np.zeros(2), 2*a1, blue), (2*a1, 2*a1+a2, orange),
                           (np.zeros(2), 2*a1+a2, purple)]:
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
ax.plot([0, 1, 3], [0, 1, 1], '--', color=gray, linewidth=1.3)
ax.text(1, -.27, '2a₁ = (2, 0)', ha='center', color=blue)
ax.text(2.75, .35, 'a₂ = (1, 1)', color=orange)
ax.text(1.05, .65, '位移 (3, 1)', color=purple)
ax.text(.05, 1.3, '在基 (a₁, a₂) 下，坐标是 (2, 1)')
ax.set(xlim=(-.2, 4.2), ylim=(-.5, 1.7), xlabel='水平位移（m）', ylabel='竖直位移（m）', title='矩阵的列决定方向，坐标决定各取多少')
ax.set_aspect('equal'); ax.grid()
save(fig, 'basis-combination')

# 零空间不是“没有状态”，而是某个状态变化不会改变这次观测。
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
velocities = np.array([0., 1., 2.])
axes[0].plot([1, 1], [-.3, 2.3], '--', color=gray)
axes[0].scatter(np.ones(3), velocities, c=[blue, orange, purple], s=85)
axes[0].annotate('速度可以改变\n位置仍是 1 m', xy=(1, 1), xytext=(.1, 1.65), arrowprops=dict(arrowstyle='->', color=gray))
axes[0].set(xlim=(0, 2), ylim=(-.4, 2.5), xlabel='状态中的位置 p（m）', ylabel='状态中的速度 v（m/s）', title='输入：三个不同状态')
axes[1].plot([0, 2], [0, 2], '--', color=gray, label='可得到的观测：y₁ = y₂')
axes[1].scatter([1], [1], color=orange, s=110, zorder=3)
axes[1].annotate('三个状态都得到 (1, 1)', xy=(1, 1), xytext=(.1, 1.65), arrowprops=dict(arrowstyle='->', color=gray))
axes[1].set(xlim=(0, 2), ylim=(0, 2), xlabel='第一台的位置读数 y₁（m）', ylabel='第二台的位置读数 y₂（m）', title='输出：H = [[1, 0], [1, 0]]')
axes[1].legend(loc='lower right'); axes[1].set_aspect('equal')
save(fig, 'measurement-nullspace')

# 把两种误差分别除以容许尺度，再画单位球；避免给混合物理单位画圆。
fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout='constrained')
theta = np.linspace(0, 2*np.pi, 401)
curves = [([-1, 0, 1, 0, -1], [0, 1, 0, -1, 0]),
          (np.cos(theta), np.sin(theta)), ([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1])]
for ax, (x, y), title, color in zip(axes, curves, ['L₁：绝对值之和 = 1', 'L₂：欧氏长度 = 1', 'L∞：最大绝对分量 = 1'], [blue, orange, purple]):
    ax.fill(x, y, color=color, alpha=.1); ax.plot(x, y, color=color)
    ax.axhline(0, color=gray, lw=.7); ax.axvline(0, color=gray, lw=.7)
    ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), xticks=[-1, 0, 1], yticks=[-1, 0, 1], xlabel='位置误差 / 位置尺度', ylabel='速度误差 / 速度尺度', title=title)
    ax.set_aspect('equal')
save(fig, 'norm-balls')

# 标量函数是雅可比的一维特例；误差严格等于扰动的平方。
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
x = np.linspace(-.2, 2.2, 300)
axes[0].plot(x, x*x, label='函数 y = x²', color=blue)
axes[0].plot(x, 1+2*(x-1), '--', label='在 x = 1 处线性化', color=orange)
axes[0].scatter([1], [1], color=gray, zorder=3)
axes[0].axvspan(.8, 1.2, color=gray, alpha=.1)
axes[0].set(xlabel='输入 x（无量纲）', ylabel='输出 y（无量纲）', title='切线描述基点附近的变化'); axes[0].legend()
delta = np.linspace(-1, 1, 300)
axes[1].plot(delta, delta**2, color=purple)
axes[1].scatter([-.2, .2], [.04, .04], color=orange)
axes[1].annotate('扰动 ±0.2 → 误差 0.04', xy=(.2, .04), xytext=(-.65, .55), arrowprops=dict(arrowstyle='->'))
axes[1].set(xlabel='输入扰动 Δx（无量纲）', ylabel='函数值 − 线性近似值', title='此例的遗漏项恰为 (Δx)²')
save(fig, 'local-linearization')

# 1.2：不同分布放在相邻面板中，纵轴分别明确为概率和密度。
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
axes[0].vlines([-.2, 0, .2], 0, [.25, .5, .25], color=blue, lw=3)
axes[0].scatter([-.2, 0, .2], [.25, .5, .25], color=blue, s=65)
for x, p in zip([-.2, 0, .2], [.25, .5, .25]): axes[0].text(x, p+.025, f'{p:g}', ha='center')
axes[0].set(xlim=(-.3, .3), ylim=(0, .65), xlabel='离散误差 e（m）', ylabel='单点概率 P(E = e)', title='三点模型：把各点的概率相加')
axes[1].plot([-.14, -.1, -.1, .1, .1, .14], [0, 0, 5, 5, 0, 0], color=blue)
axes[1].fill_between([-.02, .02], 0, 5, color=orange, alpha=.35)
axes[1].annotate('阴影面积 = 5 × 0.04 = 0.2', xy=(0, 2), xytext=(-.13, 6), arrowprops=dict(arrowstyle='->'))
axes[1].set(xlim=(-.15, .15), ylim=(0, 7), xticks=[-.1, 0, .1], xlabel='连续误差 e（m）', ylabel='概率密度 f(e)（1/m）', title='均匀模型：把区间上的面积相加')
save(fig, 'mass-and-density')

# 两图具有完全相同的边缘分布，并且协方差都是零；联合分布却不同。
# 独立图每列的概率为 1/3，每行分别为 1/3、2/3。
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
for ax, dependent in zip(axes, [False, True]):
    for x in [-1, 0, 1]:
        for y in [0, 1]:
            p = (1/3 if y == x*x else 0) if dependent else (1/9 if y == 0 else 2/9)
            if p:
                ax.scatter([x], [y], s=1100*p, color=orange if dependent else blue, zorder=3)
                ax.text(x, y+.16, '1/3' if dependent else ('1/9' if y == 0 else '2/9'), ha='center')
    ax.set(xlim=(-1.5, 1.5), ylim=(-.35, 1.55), xticks=[-1, 0, 1], yticks=[0, 1], xlabel='X（无量纲）', ylabel='Y（无量纲）', title='Y = X²：依赖但不相关' if dependent else '独立：每一列的条件比例相同')
    ax.grid()
save(fig, 'zero-correlation')

fig, ax = plt.subplots(figsize=(8, 4.3), layout='constrained')
w = np.linspace(0, 1, 301)
variance = .2**2*w**2 + .4**2*(1-w)**2
ax.plot(w, variance, color=blue)
for weight, label, xytext in [(.5, '等权：0.050 m²', (.08, .065)), (.8, '最优：0.032 m²', (.48, .125)), (1., '只用第一台：0.040 m²', (.58, .085))]:
    v = .04*weight**2+.16*(1-weight)**2
    ax.scatter([weight], [v], color=orange, zorder=3)
    ax.annotate(label, xy=(weight, v), xytext=xytext, arrowprops=dict(arrowstyle='->', color=gray))
ax.set(xlim=(0, 1.04), ylim=(0, .18), xlabel='第一台传感器的权重 w（第二台为 1 − w）', ylabel='融合误差的方差（m²）', title='独立、无偏误差：调权重就是在这条曲线上选点')
save(fig, 'fusion-weights')

# 1.3：使用解析方差，不用一次随机试验来“证明”偏差与方差分解。
fig, ax = plt.subplots(figsize=(8, 4.3), layout='constrained')
n = np.arange(1, 101)
ax.plot(n, .04/n+.09, color=blue, label='均方误差 = 0.2²/n + 0.3²')
ax.plot(n, .04/n, '--', color=orange, label='平均值的方差 = 0.2²/n')
ax.axhline(.09, ls=':', color=purple, label='固定偏差平方 = 0.3²')
ax.set(xlabel='独立读数的个数 n', ylabel='误差平方的期望或方差（m²）', title='多测几次能压低波动，固定偏差仍然保留', ylim=(0, .15))
ax.legend(loc='upper right'); ax.grid()
save(fig, 'bias-and-averaging')

# 拟合参数来自章节的可运行短例；右图放大残差，避免在位置图上看不清。
fit = examples['fit_line']()['ordinary']
t = np.array([0., 1., 2.]); y = np.array([1.1, 2.9, 5.2])
yhat = fit['initial_position_m'] + fit['speed_m_s']*t
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
line_t = np.linspace(-.1, 2.1, 100)
axes[0].plot(line_t, fit['initial_position_m']+fit['speed_m_s']*line_t, color=blue, label='拟合轨迹')
axes[0].scatter(t, y, color=orange, label='位置观测', zorder=3)
axes[0].vlines(t, yhat, y, color=orange, linestyle='--')
axes[0].set(xlabel='时间 t（s）', ylabel='位置（m）', title='p̂(t) ≈ 1.0167 + 2.05t'); axes[0].legend()
residual = y-yhat
axes[1].axhline(0, color=gray, lw=1)
axes[1].vlines(t, 0, residual, color=orange, lw=2)
axes[1].scatter(t, residual, color=orange)
for ti, r in zip(t, residual): axes[1].text(ti, r+(.015 if r>0 else -.03), f'{r:.4f}', ha='center')
axes[1].set(xlim=(-.3, 2.3), ylim=(-.23, .15), xticks=t, xlabel='时间 t（s）', ylabel='观测 − 拟合值（m）', title='把竖直差值单独放大，才看得清残差')
save(fig, 'fit-and-residuals')

# 1.4：先验与报警后的后验位于风险曲线交点两侧。
fig, ax = plt.subplots(figsize=(8.5, 4.4), layout='constrained')
q = np.linspace(0, .18, 200)
ax.plot(100*q, 100*q, color=blue, label='继续：100q')
ax.axhline(2, color=orange, ls='--', label='检查：2')
ax.axvline(2, color=gray, ls=':', lw=1.5)
ax.text(2.5, 15, '超过 2% 后，检查的期望代价更低')
ax.scatter([1, 100*90/585], [1, 2], color=purple, zorder=3)
ax.annotate('先验 1%：继续', xy=(1, 1), xytext=(.4, 7), arrowprops=dict(arrowstyle='->', color=gray))
ax.annotate('报警后 15.4%：检查', xy=(100*90/585, 2), xytext=(8.3, 6), arrowprops=dict(arrowstyle='->', color=gray))
ax.set(xlim=(0, 18), ylim=(0, 19), xlabel='异常概率 q（%）', ylabel='期望代价（统一教学单位）', title='同一个概率，换成行动后果来比较'); ax.legend(loc='upper left')
save(fig, 'decision-threshold')

# 条件风险必须按分支出现的概率加权；右图明确不含检测本身的费用。
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained', gridspec_kw={'width_ratios': [1.35, 1]})
ax = axes[0]; ax.set(xlim=(0, 1), ylim=(0, 1), title='观察结果，再分别选动作'); ax.axis('off')
ax.text(.02, .5, '做一次\n检测', va='center', ha='left')
for y, probability, action, cost, color in [(.78, '报警：0.0585', '检查', '2', orange), (.2, '不报警：0.9415', '继续', '100 × 10/9415', blue)]:
    ax.annotate('', xy=(.57, y), xytext=(.18, .5), arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax.text(.28, y+(.07 if y>.5 else -.08), probability, ha='left', color=color)
    ax.text(.6, y, f'{action}\n条件期望代价\n{cost}', va='center')
ax.text(.03, .98, '加权后：0.0585 × 2 + 0.9415 × (100 × 10/9415)', va='top', fontsize=12)
ax = axes[1]
ax.bar(['不检测', '检测后决策'], [1, .217], color=[gray, blue], width=.5)
for i, value in enumerate([1, .217]): ax.text(i, value+.03, f'{value:g}', ha='center')
ax.text(.98, .97, '可节省 0.783', transform=ax.transAxes, ha='right', va='top', color=blue)
ax.set(ylim=(0, 1.2), ylabel='期望代价（统一教学单位）', title='先不计检测费用，比较两种流程')
save(fig, 'value-of-information')

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
a = np.linspace(0, 10, 401)
for ax, risk, optimum, unit, title in [
    (axes[0], (2/3)*(a-2)**2+(1/3)*(a-8)**2, 4, 's²', '平方损失：均值 4 s 最优'),
    (axes[1], (2/3)*abs(a-2)+(1/3)*abs(a-8), 2, 's', '绝对损失：中位数 2 s 最优')]:
    ax.plot(a, risk, color=blue)
    value = np.interp(optimum, a, risk)
    ax.scatter([optimum], [value], color=orange, zorder=3)
    ax.axvline(optimum, color=orange, ls=':', lw=1.5)
    ax.set(xlabel='报告的时间 a（s）', ylabel=f'期望损失（{unit}）', title=title)
    ax.grid()
save(fig, 'loss-and-estimate')

# 1.5：同一目标与初值，只更改步长；画带符号误差以保留越过最优点的信息。
fig, ax = plt.subplots(figsize=(8.5, 4.3), layout='constrained')
k = np.arange(9)
for alpha, color, style, marker in [(.5, blue, '-', 'o'), (1.5, orange, '--', 's'), (2.2, purple, ':', '^')]:
    error = -2*(1-alpha)**k
    ax.plot(k, error, color=color, linestyle=style, marker=marker, label=f'步长 α = {alpha:g}')
ax.axhline(0, color=gray, lw=1)
ax.set(xlabel='更新次数 k', ylabel='离最优点的带符号误差 u(k) − 2', title='同从 u₀ = 0 出发：靠近、来回靠近、越跳越远', xticks=k)
ax.legend(loc='upper left', ncol=3); ax.grid()
save(fig, 'gradient-step-sizes')

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
u = np.linspace(-.4, 4.4, 301)
axes[0].plot(u, .5*(u-2)**2, color=blue, label='J(u) = 0.5(u − 2)²')
axes[0].plot([0, 4], [2, 2], '--', color=orange, label='连接两点的弦')
axes[0].scatter([2], [0], color=orange)
axes[0].set(xlabel='u（无量纲）', ylabel='J(u)', title='凸函数：弦位于曲线上方'); axes[0].legend()
u = np.linspace(-1.65, 1.65, 301)
axes[1].plot(u, (u*u-1)**2, color=blue, label='J(u) = (u² − 1)²')
axes[1].plot([-1, 1], [0, 0], '--', color=orange, label='这条弦落到曲线下方')
axes[1].scatter([-1, 0, 1], [0, 1, 0], color=orange)
axes[1].annotate('导数为零，却是局部最大', xy=(0, 1), xytext=(-1.5, 1.8), arrowprops=dict(arrowstyle='->', color=gray))
axes[1].set(xlabel='u（无量纲）', ylabel='J(u)', title='非凸函数：驻点未必是最小值'); axes[1].legend(loc='upper center')
save(fig, 'convex-and-stationary')

# 1.6：所有节点的值直接来自教程里的求解器；橙色仅标一条执行轨迹。
values, _ = dp['backward_induction'](1.)
fig, ax = plt.subplots(figsize=(9.5, 5.8), layout='constrained')
for time in range(2):
    for state in range(3):
        for next_state in ([state, state+1] if state<2 else [state]):
            highlighted = state == time and next_state == state+1
            ax.annotate('', xy=(time+1, next_state), xytext=(time, state),
                        arrowprops=dict(arrowstyle='->', shrinkA=35, shrinkB=35,
                                        color=orange if highlighted else gray,
                                        lw=3 if highlighted else 1.2,
                                        linestyle='-' if next_state!=state else '--'))
for time in range(3):
    for state in range(3):
        ax.text(time, state, f's = {state}\nV = {values[time][state]:g}', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=.45', facecolor='white', edgecolor=blue, linewidth=1.5))
ax.annotate('执行轨迹：从左到右，前进两次，总代价 2', xy=(1.95, - .55), xytext=(.05, -.55), ha='left', arrowprops=dict(arrowstyle='->', color=orange))
ax.annotate('求值：从终点向左倒推', xy=(.05, 2.65), xytext=(1.95, 2.65), ha='right', arrowprops=dict(arrowstyle='->', color=blue))
ax.text(0, 3.08, '虚线：停留，代价 0；实线：前进，代价 1', color=gray)
ax.set(xlim=(-.5, 2.5), ylim=(-.8, 3.3), xticks=[0, 1, 2], xticklabels=['k = 0', 'k = 1', 'k = 2（终点）'], yticks=[], xlabel='时间层；s 是站点编号，V 使用统一教学代价单位')
for spine in ax.spines.values(): spine.set_visible(False)
save(fig, 'dp-time-layers')

random_values, _ = dp['backward_induction'](.8)
fig, ax = plt.subplots(figsize=(9, 4.7), layout='constrained')
ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
ax.text(.04, .52, 'k = 1，s = 1\n尝试前进\n本步先付出 1', ha='left', va='center', bbox=dict(boxstyle='round,pad=.6', facecolor='white', edgecolor=blue))
for y, label, end, color in [(.8, '成功：0.8', 'k = 2，s = 2\n终点代价 0', blue), (.26, '失败：0.2', 'k = 2，s = 1\n终点代价 4', orange)]:
    ax.annotate('', xy=(.72, y), xytext=(.3, .52), arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax.text(.43, y+.05, label, color=color, ha='center')
    ax.text(.76, y, end, ha='left', va='center')
ax.text(.04, .98, '先求这个动作的期望代价，再和“停留”的代价 4 比较', va='top')
ax.text(.12, .04, f'前进：1 + 0.8 × 0 + 0.2 × 4 = {random_values[1][1]:g} < 4，因此选前进')
save(fig, 'dp-random-branches')
