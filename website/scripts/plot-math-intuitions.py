"""生成数学入门配图（教学设定和解析曲线，非实测结果）。

python scripts/plot-math-intuitions.py --font /path/to/chinese-font.ttf
使用已有 NumPy/Matplotlib；SVG 保存字形路径，不依赖读者的中文字体。
"""
import argparse
from pathlib import Path
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
font = FontProperties(fname=args.font)
plt.style.use(Path(__file__).with_name('teaching.mplstyle'))
blue, orange, gray, _ = plt.rcParams['axes.prop_cycle'].by_key()['color']


def save(fig, name):
    """中文字体也应用于刻度；PNG用于本地核对，网页使用SVG。"""
    for index, ax in enumerate(fig.axes):
        # 面板字母按阅读顺序编号，正文可准确指向同一幅图的局部。
        if len(fig.axes) > 1:
            ax.set_title(f'({chr(97 + index)}) {ax.get_title()}', loc='center')
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]:
            size = max(12, label.get_fontsize())
            label.set_fontproperties(font)
            label.set_fontsize(size)
        if ax.get_legend():
            for label in ax.get_legend().get_texts():
                size = max(12, label.get_fontsize())
                label.set_fontproperties(font)
                label.set_fontsize(size)
    fig.savefig(args.output / (name + '.svg'), bbox_inches='tight')
    fig.savefig(args.output / (name + '.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)


# 条件概率改变计数的样本集合，两幅图分别归一化。
fig, axes = plt.subplots(2, 1, figsize=(8.5, 4.2), layout='constrained')
for ax, fraction, title, note in zip(axes, [100/10000, 90/585],
        ['全部 10000 个样本', '仅看发出报警的 585 个样本'],
        ['异常 100 / 10000 = 1.0%', '异常 90 / 585 ≈ 15.4%']):
    ax.barh(0, 100*fraction, color=orange, height=.5, label='实际异常')
    ax.barh(0, 100*(1-fraction), left=100*fraction, color=gray, height=.5, label='实际正常')
    ax.set(xlim=(0, 100), ylim=(-.6, .6), yticks=[], xlabel='各自样本集合中的比例（%）', title=title)
    ax.text(0, .36, note, fontsize=12)
axes[0].legend(loc='lower right', ncol=2)
save(fig, 'conditioning')

# 独立的 N(10,1) 测量平均后方差为 1/n；保持三个密度的面积均为1。
fig, ax = plt.subplots(figsize=(8.5, 4.2), layout='constrained')
x = np.linspace(6, 14, 1001)
for n, color, style in [(1, gray, ':'), (4, blue, '--'), (25, orange, '-')]:
    sigma = 1 / np.sqrt(n)
    density = np.exp(-.5*((x-10)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
    ax.plot(x, density, color=color, linestyle=style, linewidth=2.2, label=f'n = {n}，标准差 = {sigma:.1f} m')
ax.set(xlabel='平均读数（m）', ylabel='概率密度（1/m）', title='独立、无偏高斯测量：重复取 n 个读数后求平均')
ax.axvline(10, color=gray, linestyle=':', linewidth=1)
ax.legend(loc='upper right')
save(fig, 'sample-mean')

# 同一个二次目标，对照梯度更新和可行域限制。
fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout='constrained')
u = np.linspace(-1.3, 3.1, 500)
for ax in axes:
    ax.plot(u, .5*(u-2)**2, color=blue, linewidth=2)
    ax.set(xlabel='无量纲控制量 u', ylabel='目标 J(u)', xlim=(-1.3, 3.1), ylim=(-.2, 5.6))
steps = np.array([0., 1., 1.5, 1.75])
axes[0].scatter(steps, .5*(steps-2)**2, color=orange, zorder=3)
for i in range(3):
    axes[0].annotate('', xy=(steps[i+1], .5*(steps[i+1]-2)**2), xytext=(steps[i], .5*(steps[i]-2)**2),
                     arrowprops={'arrowstyle': '->', 'color': orange})
axes[0].set_title('无约束：步长 0.5，依次更新')
axes[0].text(-.6, 4.7, 'u：0 → 1 → 1.5 → 1.75')
axes[1].axvspan(-1, 1, color=blue, alpha=.1)
axes[1].axvline(1, color=orange, linestyle=':')
axes[1].scatter([1, 2], [.5, 0], c=[orange, gray], zorder=3)
axes[1].text(-.9, 4.7, '允许范围：−1 ≤ u ≤ 1')
axes[1].annotate('最优可行动作 u = 1', xy=(1, .5), xytext=(.2, 3.1), arrowprops={'arrowstyle': '->'})
axes[1].set_title('加上约束：最优解可以在边界')
save(fig, 'optimization')
print(f'Generated three analytic teaching figures in {args.output}')

# 同一个协方差例贯穿特征方向、标准化、白化与线性传播。
# 椭圆满足 e^T Σ^-1 e = 1，是等密度线，不标作 95% 概率区域。
theta = np.linspace(0, 2*np.pi, 300)
circle = np.array([np.cos(theta), np.sin(theta)])
covariance = np.array([[4., 1.2], [1.2, 1.]])
ellipse = np.linalg.cholesky(covariance) @ circle
fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
ax = axes[0, 0]
ax.plot([0, 2.3], [0, 2.3], color=gray, label='共同位置模型 (p, p)')
ax.quiver([0, 0], [0, 0], [1, 1.5], [2, 1.5], angles='xy', scale_units='xy', scale=1,
          color=[orange, blue], width=.008)
ax.plot([1.5, 1], [1.5, 2], '--', color=orange, label='残差与模型方向垂直')
ax.text(.55, 2.12, '观测 (1, 2)')
ax.text(1.5, 1.2, '拟合 (1.5, 1.5)')
ax.set(xlim=(0, 2.5), ylim=(0, 2.5), xlabel='第一个读数（m）', ylabel='第二个读数（m）', title='两次等精度观测：最小二乘投影')
ax.legend(loc='lower right', fontsize=9)
ax = axes[0, 1]
ax.plot(*ellipse, color=blue)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
for val, direction in zip(eigenvalues, eigenvectors.T):
    tip = np.sqrt(val) * direction
    ax.plot([-tip[0], tip[0]], [-tip[1], tip[1]], '--', color=orange)
ax.set(xlabel='位置误差 e₁（m）', ylabel='位置误差 e₂（m）', title='相关误差：长短轴沿特征方向')
ax = axes[1, 0]
standardized = np.diag(1 / np.sqrt(np.diag(covariance))) @ ellipse
whitening = np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T
ax.plot(*standardized, color=blue, label='标准化：相关仍为 0.6')
ax.plot(*(whitening @ ellipse), '--', color=orange, label='白化：协方差为单位阵')
ax.set(xlabel='变换后分量 1（无量纲）', ylabel='变换后分量 2（无量纲）', title='缩放各轴与消除相关的区别')
ax.set_ylim(-1.2, 1.65)
ax.legend(loc='upper right', fontsize=9)
ax = axes[1, 1]
transform = np.array([[1., 1.], [0., 1.]])
ax.plot(*ellipse, color=blue, label='变换前')
ax.plot(*(transform @ ellipse), '--', color=orange, label='剪切后：AΣA^T')
ax.set(xlabel='位置误差分量 1（m）', ylabel='位置误差分量 2（m）', title='线性传播改变误差形状', ylim=(-1.2, 1.8))
ax.legend(loc='upper left', fontsize=9)
for ax in axes.flat:
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.15)
save(fig, 'measurement-geometry')

# 复用零号工程生成的数据，不再维护另一份采样与拟合算法。
import runpy
examples = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'ngc/examples/math_tools.py'))
mean_error, reading_error, summary = examples['repeated_predictions']()
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
for ax, errors, sd, title in zip(axes, [mean_error, reading_error],
        [summary['mean_prediction_sd_theory_m'], summary['new_reading_error_sd_theory_m']],
        ['拟合均值 − 真实均值', '新的独立读数 − 拟合均值']):
    ax.hist(errors, bins=np.linspace(-1.5, 1.5, 36), density=True, color=blue, alpha=.45, label='2000 次重复拟合')
    x = np.linspace(-1.5, 1.5, 400)
    ax.plot(x, np.exp(-.5*(x/sd)**2)/(sd*np.sqrt(2*np.pi)), color=orange, label=f'理论标准差 {sd:.4f} m')
    ax.set(title=title, xlabel='t = 3 s 处的误差（m）', ylabel='概率密度（1/m）', xlim=(-1.5, 1.5), ylim=(0, 1.75))
    ax.legend(loc='upper right', fontsize=9)
    ax.text(.02, .96, 'σ = 0.2 m，seed = 7', transform=ax.transAxes, va='top', fontsize=9)
save(fig, 'estimation-uncertainty')

# 二维狭长谷地；相同步长沿不同曲率方向具有不同收缩因子。
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout='constrained')
u, v = np.meshgrid(np.linspace(-1.5, 1.5, 300), np.linspace(-1.2, 1.2, 300))
axes[0].contour(u, v, .5*(u**2+9*v**2), levels=[.1, .3, .7, 1.5, 3, 5], colors=gray, linewidths=.8)
steps = [np.array([1., 1.])]
for _ in range(14):
    steps.append(steps[-1] - .18 * np.array([steps[-1][0], 9*steps[-1][1]]))
steps = np.array(steps)
axes[0].plot(*steps.T, '-o', color=orange, markersize=3)
axes[0].set(title='J = ½(u² + 9v²)：步长 0.18', xlabel='u（无量纲）', ylabel='v（无量纲）')
axes[0].text(-1.4, 1.16, 'v 交替变号，u 缓慢靠近零', fontsize=12)
axes[0].set_ylim(-1.2, 1.35)
axes[1].fill([0, 1, 0], [0, 0, 1], color=blue, alpha=.15, label='u₁ + u₂ ≤ 1，u₁,u₂ ≥ 0')
axes[1].plot([0, 1], [1, 0], color=blue)
axes[1].scatter([1, .5], [1, .5], c=[orange, blue])
axes[1].annotate('候选 (1, 1) 不可行', xy=(1, 1), xytext=(.05, 1.2), arrowprops={'arrowstyle': '->'})
axes[1].annotate('最近可行点 (0.5, 0.5)', xy=(.5, .5), xytext=(.05, -.24), arrowprops={'arrowstyle': '->'})
axes[1].plot([1, .5], [1, .5], '--', color=orange)
axes[1].set(title='耦合约束：逐分量截到 [0, 1] 不够', xlabel='u₁（无量纲）', ylabel='u₂（无量纲）', xlim=(-.15, 1.35), ylim=(-.35, 1.4))
axes[1].legend(loc='upper right', bbox_to_anchor=(1, 1.03), fontsize=9)
for ax in axes:
    ax.set_aspect('equal', adjustable='box')
save(fig, 'optimization-geometry')
print('Generated three additional figures; sampling figure reuses examples/math_tools.py')
