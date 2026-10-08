"""Original figures for sections 4.5–4.7.

From website/: python scripts/generate-flight-aircraft-figures.py \
    --font '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
Requires NumPy and Matplotlib; uses scripts/teaching.mplstyle. The public copy
contains the same source. All parameters below are teaching assumptions.
"""
from pathlib import Path
import argparse
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
import numpy as np

BLUE, ORANGE, GRAY = '#2563a6', '#bc5930', '#6e7b86'


def save(fig, folder, name):
    for extension in ('svg', 'png'):
        fig.savefig(folder / f'aircraft-{name}.{extension}', bbox_inches='tight')
    plt.close(fig)


def arrow(ax, start, end, color=GRAY, style='-'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2, linestyle=style))


def actuator(folder):
    # Exact solution: 10-degree step, tau=.1 s, slew limit=40 deg/s.
    t = np.linspace(0, .6, 301)
    command_deg, tau_s, rate_deg_s = 10., .1, 40.
    transition = (command_deg - tau_s*rate_deg_s)/rate_deg_s
    limited = np.where(t <= transition, rate_deg_s*t,
                       command_deg - tau_s*rate_deg_s*np.exp(-(t-transition)/tau_s))
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.plot(t, command_deg*(1-np.exp(-t/tau_s)), color=BLUE, label='一阶响应')
    ax.plot(t, limited, '--', color=ORANGE, label='一阶响应 + 限速')
    ax.axhline(10, color=GRAY, ls=':', label='指令 10°')
    at_one_tau = float(np.interp(.1, t, limited))
    ax.plot([.1], [at_one_tau], 's', color=ORANGE)
    ax.annotate(f'0.1 s：{at_one_tau:.2f}°', xy=(.1,at_one_tau), xytext=(.21,3.2),
                arrowprops=dict(arrowstyle='-', color=GRAY))
    ax.set(xlabel='时间 / s', ylabel='实际舵偏 / °', xlim=(0,.6), ylim=(0,11))
    ax.grid(True)
    ax.legend(loc='lower right')
    save(fig, folder, 'actuator-step')


def fixed_wing(folder):
    # Side view: horizontal +x_b, screen-up is -z_b.
    a = np.deg2rad(20)
    velocity = np.array([np.cos(a), -np.sin(a)])
    lift = np.array([np.sin(a), np.cos(a)])
    fig, ax = plt.subplots(figsize=(7.4,4.7), constrained_layout=True)
    arrow(ax, (-1.15,0), (2.5,0))
    arrow(ax, (0,1.4), (0,-1.2))
    ax.text(2.5,.08, r'$x_b$ 前', ha='right')
    ax.text(.06,-1.22, r'$z_b$ 下', va='top')
    arrow(ax, (0,0), 2.3*velocity, BLUE)
    ax.text(2.0,-.94, r'$\mathbf{v}_{a,b}$：沿风轴 $+x_w$', color=BLUE, ha='center')
    arrow(ax, (0,0), 1.2*lift, BLUE)
    ax.text(.45,1.18, r'$L$：垂直空速', color=BLUE)
    arrow(ax, (0,0), -1.05*velocity, ORANGE, '--')
    ax.text(-1.06,.48, r'$D$：反向空速', color=ORANGE, ha='center')
    theta = np.linspace(-a,0,40)
    ax.plot(.65*np.cos(theta), .65*np.sin(theta), color=GRAY)
    ax.text(.81,-.17, r'$\alpha$', fontsize=15)
    ax.plot(0,0,'o',color='#192b35')
    ax.text(-.12,-.18,'质心',ha='right')
    ax.text(.95,1.58,'侧视：机头向右，迎角取 20°',ha='center')
    ax.text(.95,-1.55,'箭头表示方向，长度未按力或速度比例绘制',ha='center',color=GRAY)
    ax.set(xlim=(-1.7,2.65), ylim=(-1.7,1.75), aspect='equal')
    ax.axis('off')
    save(fig, folder, 'fixed-wing-forces')


def rotorcraft(folder):
    # Screen x = body y; screen y = body x; body +z points into the page.
    centers = [(0,1.15), (1.15,0), (0,-1.15), (-1.15,0)]
    fig, ax = plt.subplots(figsize=(7.4,6.1), constrained_layout=True)
    ax.plot([0,0],[-1.15,1.15], color=GRAY, lw=3)
    ax.plot([-1.15,1.15],[0,0], color=GRAY, lw=3)
    for i, center in enumerate(centers):
        ccw = i % 2 == 0
        color = BLUE if ccw else ORANGE
        ax.add_patch(Circle(center,.38,facecolor='white',edgecolor=color,lw=2,
                            linestyle='-' if ccw else '--'))
        theta = np.deg2rad(np.linspace(30,290,90) if ccw else np.linspace(290,30,90))
        curve = np.asarray(center)[:,None] + .31*np.array([np.cos(theta),np.sin(theta)])
        ax.plot(*curve,color=color,ls='-' if ccw else '--')
        arrow(ax,curve[:,-6],curve[:,-1],color)
        ax.text(*center,str(i+1),ha='center',va='center',fontsize=16,color=color)
    labels = [(0,1.72,'1 前：逆时针；机体 +N'),(1.75,.60,'2 右：顺时针\n机体 −N'),
              (0,-1.82,'3 后：逆时针；机体 +N'),(-1.75,.60,'4 左：顺时针\n机体 −N')]
    for x,y,label in labels: ax.text(x,y,label,ha='center',va='center',fontsize=12)
    arrow(ax,(0,0),(0,.73),GRAY)
    arrow(ax,(0,0),(.73,0),GRAY)
    ax.text(.11,.59,r'$+x_b$ 前',fontsize=12)
    ax.text(.23,-.19,r'$+y_b$ 右',fontsize=12)
    ax.text(-.13,-.23,'⊗',ha='center',fontsize=20)
    ax.text(-.35,-.55,r'$+z_b$ 向纸内',ha='center',fontsize=12)
    ax.text(1.6,-1.17,r'$\ell=0.20\ \mathrm{m}$',ha='center')
    ax.text(0,2.12,'上方俯视：四桨推力均指向纸外',ha='center',fontsize=13)
    ax.set(xlim=(-2.6,2.6),ylim=(-2.05,2.3),aspect='equal')
    ax.axis('off')
    save(fig, folder, 'quadrotor-layout')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', required=True, type=Path)
    args = parser.parse_args()
    website = Path.cwd()
    plt.style.use(website / 'scripts/teaching.mplstyle')
    font_manager.fontManager.addfont(args.font)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=args.font).get_name()
    folder = website / 'public/figures/flight-models'
    folder.mkdir(parents=True,exist_ok=True)
    actuator(folder)
    fixed_wing(folder)
    rotorcraft(folder)
    public_source = website / 'public/figures/sources/generate-flight-aircraft-figures.py'
    public_source.parent.mkdir(parents=True,exist_ok=True)
    if Path(__file__).resolve() != public_source.resolve():
        shutil.copyfile(__file__,public_source)
    print('Generated three SVG/PNG figures and the public source copy.')


if __name__ == '__main__':
    main()
