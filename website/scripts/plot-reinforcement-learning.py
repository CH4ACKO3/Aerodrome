"""第三章示意图与数值图；数值结果由章节程序产生，不重复实现算法。"""
import argparse
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'website/public/figures/reinforcement-learning'


def save(fig, name, font):
    """中文字体固化为 SVG 字形；同时输出 PNG 便于检查标注。"""
    for ax in fig.axes:
        labels = [ax.title, ax.xaxis.label, ax.yaxis.label,
                  *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]
        if ax.get_legend():
            labels += ax.get_legend().get_texts()
        for label in labels:
            size = max(12, label.get_fontsize())
            label.set_fontproperties(font)
            label.set_fontsize(size)
    fig.savefig(OUTPUT / f'{name}.svg', bbox_inches='tight')
    fig.savefig(OUTPUT / f'{name}.png', bbox_inches='tight', dpi=140)
    plt.close(fig)


def representation_diagram(font):
    """教学结构示意：相同的函数拟合工具承载不同的输出与目标。"""
    fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
    ax.set(xlim=(0, 10), ylim=(-.45, 3.4))
    ax.axis('off')
    blue, orange, gray, purple = plt.rcParams['axes.prop_cycle'].by_key()['color']
    rows = [
        (2.6, '状态、输入', r'动力学模型 $f_\psi$', '下一状态', '转移记录中的下一状态', blue),
        (1.35, '状态', r'值函数 $V_w$', '未来代价', '轨迹累计代价 / Bellman 目标', orange),
        (.1, '状态', r'策略 $\pi_\theta$', '控制输入', '专家动作 / 闭环性能目标', purple),
    ]
    for y, inputs, model, outputs, target, color in rows:
        ax.text(.05, y, inputs, va='center', fontsize=13)
        ax.annotate('', xy=(2.15, y), xytext=(1.55, y),
                    arrowprops=dict(arrowstyle='->', color=gray, lw=1.5))
        ax.add_patch(FancyBboxPatch((2.2, y-.28), 2.5, .56,
                                   boxstyle='round,pad=.04', ec=color, fc='white', lw=1.8))
        ax.text(3.45, y, model, va='center', ha='center', color=color, fontsize=14)
        ax.annotate('', xy=(5.6, y), xytext=(4.8, y),
                    arrowprops=dict(arrowstyle='->', color=gray, lw=1.5))
        ax.text(5.7, y, outputs, va='center', fontsize=13)
        ax.text(2.2, y-.54, '参数依据：' + target, va='center', color='#536b76', fontsize=12)
    ax.text(.05, 3.2, '输入', color=gray)
    ax.text(2.2, 3.2, '带参数的函数', color=gray)
    ax.text(5.7, 3.2, '输出的含义', color=gray)
    save(fig, 'parameter-roles', font)


def control_figures(font):
    # Reuse the numerical module, including its simulator and cost definition.
    sys.path.insert(0, str(ROOT / 'ngc/examples'))
    import dp_control as dp
    blue, orange, gray, purple = plt.rcParams['axes.prop_cycle'].by_key()['color']
    p, gain = dp.finite_lqr()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].plot(np.arange(dp.T + 1), p, color=blue, label='有限时域 P(k)')
    axes[0].axhline(.5, color=gray, ls=':', label='无限时域 P=0.5')
    axes[0].set(xlabel='时间步 k', ylabel='归一化偏差的值系数 P(k)', title='(a) 终端代价影响靠近终点的值')
    axes[1].plot(np.arange(dp.T), gain, color=orange, label='有限时域 K(k)')
    axes[1].axhline(2, color=gray, ls=':', label='无限时域 K=2')
    axes[1].set(xlabel='时间步 k', ylabel='反馈增益（s⁻¹）', title='(b) 先倒推，再按时刻执行')
    for ax in axes:
        ax.legend()
    save(fig, 'riccati-horizon', font)

    controls = [('基线 K=1', lambda x,k: dp.base_action(x), blue, '-'),
                ('一步 rollout', lambda x,k: dp.rollout_action(k,x), orange, '--'),
                ('三步候选 MPC', lambda x,k: dp.mpc_action(k,x,p), purple, '-.')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for name, policy, color, style in controls:
        trace = dp.trajectory(5., policy)
        axes[0].plot(np.arange(dp.T+1)*dp.DT, trace['x'], color=color, ls=style, label=name)
        axes[1].step(np.arange(dp.T)*dp.DT, trace['u'], where='post', color=color, ls=style, label=name)
    axes[0].set(xlabel='时间（s）', ylabel='速度偏差（m/s）', title='(a) 同一初值 x₀=5 m/s')
    axes[1].set(xlabel='时间（s）', ylabel='加速度（m/s²）', title='(b) 每步保持实际输入')
    axes[1].axhline(-dp.LIMIT, color=gray, ls=':', label='负向输入界')
    for ax in axes:
        ax.legend()
    save(fig, 'lookahead-control', font)

    data = dp.station_learning()
    steps = [h['samples'] for h in data['history']]
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.plot(steps, [h['td'][0] for h in data['history']], 'o-', color=blue, label='TD：评价始终前进')
    ax.plot(steps, [min(h['q'][0]) for h in data['history']], 's--', color=orange, label='Q 学习：最低动作价值')
    ax.axhline(data['exact_v'][0], color=gray, ls=':', label='站点0精确值 1.9')
    ax.set_xscale('symlog', linthresh=1)
    ax.set(xlim=(0, max(steps)), xlabel='每种更新的样本数（0–1 线性，其后对数）',
           ylabel='站点0的折扣代价估计', title='不同更新目标，在此例中到达同一个值')
    ax.legend()
    save(fig, 'sampled-values', font)


def policy_figures(font):
    """参数搜索、采样更新和留出评测采用各自的实际目标值。"""
    import policy_learning as policy
    blue, orange, gray, purple = plt.rcParams['axes.prop_cycle'].by_key()['color']
    search = policy.gain_search()
    reinforce = policy.reinforce_demo()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
    axes[0].plot(search['candidate_gain_per_s'], search['train_cost'], 'o-', color=blue)
    axes[0].axvline(search['selected_gain_per_s'], color=gray, ls=':', label='验证后选中的 K=2')
    axes[0].set(xlabel='反馈增益 K（s⁻¹）', ylabel='训练初值平均代价（无量纲）', title='(a) 速度控制：搜索九个增益')
    axes[0].legend()
    checkpoints = reinforce['checkpoints']
    axes[1].plot([row['episode'] for row in checkpoints], [row['prob_action1'] for row in checkpoints], 's--', color=orange)
    axes[1].set(xlabel='单步交互次数', ylabel='低代价动作的概率', ylim=(0,1.05), title='(b) 独立两动作例：REINFORCE')
    save(fig, 'policy-search-and-sampling', font)

    comparison = policy.compare_controllers(search['selected_gain_per_s'])['controllers']
    names = ['fixed_gain_0_5', 'searched_gain', 'model_lqr_clipped']
    fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
    pos = np.arange(3)
    for offset, case, label, color in [(-.18, 'nominal', '名义模型', blue), (.18, 'mismatch_and_disturbance', '输入效率变化与常值扰动', orange)]:
        bars = ax.bar(pos + offset, [comparison[case][name]['mean_cost'] for name in names], width=.36, label=label, color=color)
        ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.set(xticks=pos, xticklabels=['固定 K=0.5', '搜索 K=2', '有限时域 LQR 后限幅'], ylabel='留出初值平均代价（无量纲）', ylim=(0,35))
    ax.legend()
    save(fig, 'controller-comparison', font)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.style.use(Path(__file__).with_name('teaching.mplstyle'))
    font = FontProperties(fname=args.font)
    representation_diagram(font)
    control_figures(font)
    policy_figures(font)


if __name__ == '__main__':
    main()
