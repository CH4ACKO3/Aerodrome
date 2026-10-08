"""第二章 Q 学习与参数化控制配图，直接读取可运行例子的计算结果。"""
import argparse
from pathlib import Path
import runpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
output = root/'website/public/figures/statistical-learning'
plt.style.use(Path(__file__).with_name('teaching.mplstyle'))
font = FontProperties(fname=args.font)
examples = runpy.run_path(str(root/'ngc/examples/learning_control.py'))
blue, orange, gray, purple = plt.rcParams['axes.prop_cycle'].by_key()['color']


def save(fig, name):
    for ax in fig.axes:
        labels = [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]
        if ax.get_legend():
            labels += ax.get_legend().get_texts()
        for label in labels:
            size = max(12, label.get_fontsize())
            label.set_fontproperties(font)
            label.set_fontsize(size)
    fig.savefig(output/f'{name}.svg', bbox_inches='tight')
    fig.savefig(output/f'{name}.png', bbox_inches='tight', dpi=140)
    plt.close(fig)


d = examples['q_learning_demo']()
fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
for state, color in enumerate([blue, orange]):
    ax.plot(d['samples'], d['estimates'][:, state], color=color, label=f'站点 {state}：采样估计')
    ax.axhline(d['optimal_values'][state], color=color, linestyle='--', label=f'站点 {state}：解析最优值')
ax.set(xlabel='采样次数', ylabel='价值（代价单位）', title='前100次采样中的价值更新', xlim=(0,100), ylim=(-.05, 2.4))
ax.legend(ncols=2, loc='upper center')
save(fig, 'q-learning')

d = examples['control_demo']()
fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout='constrained')
axes[0].plot(d['gains'], d['train_costs'], 'o-', color=blue, label='训练工况均值')
axes[0].plot(d['shortlist'], d['validation_costs'], 's', color=orange, label='候选的验证均值')
axes[0].axvline(d['selected_gain'], color=gray, linestyle=':', label='选定增益')
axes[0].set(xlabel='比例增益 K（s⁻¹）', ylabel='轨迹代价（无量纲）', title='(a) 参数选择')
axes[0].legend()
for key, label, color, style in [('selected_trace',f"选定 K={d['selected_gain']:g}",blue,'-'),('baseline_trace','基线 K=0.5',orange,'--')]:
    t = d[key]
    axes[1].plot(t['time'], t['error'], color=color, linestyle=style, label=label)
axes[1].set(xlabel='时间（s）', ylabel='速度误差（m/s）', title='(b) 留出初值 e(0)=5 m/s')
axes[1].legend()
save(fig, 'parametric-control')
