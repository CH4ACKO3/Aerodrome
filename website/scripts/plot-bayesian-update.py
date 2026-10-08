"""第一章先验、似然、后验与抽样配图，复用 bayesian_update.py 的结果。"""
import argparse
from pathlib import Path
import runpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
out = root/'website/public/figures/math-tools'
font = FontProperties(fname=args.font)
plt.style.use(Path(__file__).with_name('teaching.mplstyle'))
examples = runpy.run_path(str(root/'ngc/examples/bayesian_update.py'))
blue, orange, gray, purple = plt.rcParams['axes.prop_cycle'].by_key()['color']


def save(fig, name):
    for ax in fig.axes:
        labels = [ax.title, ax.xaxis.label, ax.yaxis.label,
                  *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]
        if ax.get_legend():
            labels += ax.get_legend().get_texts()
        for label in labels:
            size = max(12, label.get_fontsize())
            label.set_fontproperties(font)
            label.set_fontsize(size)
    fig.savefig(out/f'{name}.svg', bbox_inches='tight')
    fig.savefig(out/f'{name}.png', bbox_inches='tight', dpi=140)
    plt.close(fig)


d = examples['conjugate_updates']()
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
axes[0].plot(d['theta'], d['prior_density'], color=blue, label='先验 Beta(2,2)')
axes[0].plot(d['theta'], d['posterior_density'], color=orange, ls='--', label='后验 Beta(10,4)')
axes[0].set(xlabel='成功率 θ（无量纲）', ylabel='θ 的概率密度', title='(a) 两条密度的面积各为1', xlim=(0,1), ylim=(0,None))
axes[0].legend()
axes[1].plot(d['theta'], d['relative_likelihood'], color=purple)
axes[1].axvline(.8, color=gray, ls=':', label='最大似然 θ=0.8')
axes[1].set(xlabel='假设的成功率 θ（无量纲）', ylabel='相对似然（峰值归一）',
            title='(b) 固定8次成功、2次失败的数据', xlim=(0,1), ylim=(0,1.15))
axes[1].legend()
save(fig, 'beta-update')

d = examples['sampling_demo']()
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained', sharey=True)
for ax, n, color in zip(axes, [30,3000], [blue,orange]):
    ax.hist(d['observations_m'][:n], bins=np.linspace(8,12,21), density=True,
            color=color, alpha=.28, edgecolor=color, label='归一化样本直方图')
    ax.plot(d['grid_m'], d['density_per_m'], color=gray, ls='--', label='设定的高斯密度')
    ax.set(xlabel='读数（m）', ylabel='概率密度（m⁻¹）', title=f'{n} 个样本', xlim=(8,12))
    ax.legend()
save(fig, 'distribution-sampling')
