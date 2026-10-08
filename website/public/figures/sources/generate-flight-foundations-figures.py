"""Original geometry for Chapter 4.1–4.3 (no aircraft data required).

From website/: python scripts/generate-flight-foundations-figures.py
Requires NumPy and Matplotlib. Uses teaching.mplstyle and Arial Unicode on macOS.
Outputs foundations-*.svg/.png and a public copy of this source.
"""
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/figures/flight-models'
BLUE, ORANGE, GRAY, INK = '#2563a6', '#bc5930', '#6e7b86', '#192b35'
plt.style.use(ROOT / 'scripts/teaching.mplstyle')
FONT = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
font_manager.fontManager.addfont(FONT)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    for extension in ('svg', 'png'):
        fig.savefig(OUT / f'foundations-{name}.{extension}', bbox_inches='tight')
    plt.close(fig)


def arrow(ax, start, end, color=GRAY, **kwargs):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2, **kwargs))


def flow():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set(xlim=(0, 10), ylim=(0, 5))
    ax.axis('off')
    nodes = [(1.6, 3.9, '当前状态与输入', '位置、速度、姿态；指令'),
             (5.0, 3.9, '环境与部件', '风、大气、实际舵偏 / 转速'),
             (8.4, 3.9, '机体系载荷', '合力 F；质心合力矩 M'),
             (8.4, 1.4, '状态导数', '刚体与部件的变化率'),
             (5.0, 1.4, '数值积分', '按步长推进时间'),
             (1.6, 1.4, '下一时刻', '新状态、观测与记录')]
    for x, y, title, subtitle in nodes:
        ax.add_patch(FancyBboxPatch((x-1.43, y-.58), 2.86, 1.16,
                                   boxstyle='round,pad=0.06,rounding_size=0.09',
                                   facecolor='#f7f9fa', edgecolor=GRAY, lw=1))
        ax.text(x, y+.15, title, ha='center', va='center', fontsize=14, color=INK)
        ax.text(x, y-.24, subtitle, ha='center', va='center', fontsize=12)
    for start, end in [((3.1,3.9),(3.5,3.9)), ((6.5,3.9),(6.9,3.9)),
                       ((8.4,3.25),(8.4,2.05)), ((6.9,1.4),(6.5,1.4)),
                       ((3.5,1.4),(3.1,1.4)), ((1.6,2.05),(1.6,3.25))]:
        arrow(ax,start,end)
    ax.text(2.0,2.6,'下一步', fontsize=12)
    ax.text(5.0,2.55,'积分器在子阶段重新计算\n状态相关的载荷与导数',
            ha='center', va='center', color=BLUE, fontsize=13)
    save(fig,'simulation-loop')


def yaw():
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.set(xlim=(-1.9,2.8), ylim=(-1.6,1.7), aspect='equal')
    ax.axis('off')
    arrow(ax,(-1.6,0),(2.25,0),GRAY)
    arrow(ax,(0,-1.35),(0,1.45),GRAY)
    ax.text(2.25,-.2,'东 E', ha='center')
    ax.text(-.12,1.5,'北 N', ha='right')
    arrow(ax,(0,0),(1.75,0),BLUE)
    arrow(ax,(0,0),(0,-1.1),BLUE)
    ax.text(1.1,.14,'机头 +x_b', color=BLUE, ha='center')
    ax.text(.13,-1.1,'右翼 +y_b', color=BLUE)
    arrow(ax,(0,.9),(.9,0),ORANGE,connectionstyle='arc3,rad=-0.5')
    ax.text(.7,.82,'ψ = +90°',color=ORANGE)
    ax.plot(0,0,'o',color=INK,ms=5)
    ax.text(-1.7,-.7,'俯视：下 D 与 +z_b\n均指入纸面',fontsize=12)
    ax.text(1.1,-.55,'沿机头 10 m/s\n= 向东 10 m/s',ha='center',fontsize=12)
    save(fig,'yaw')


def rotation_order():
    # Active rotations around fixed reference axes, acting on column vectors.
    rx = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    rz = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    matrices = (rz @ rx, rx @ rz)
    assert np.array_equal(matrices[0][:,0],[0,1,0])
    assert np.array_equal(matrices[1][:,0],[0,0,1])
    fig = plt.figure(figsize=(10.2,5.0))
    titles = ['(a) 先绕固定 x，再绕固定 z', '(b) 先绕固定 z，再绕固定 x']
    for i,(matrix,title) in enumerate(zip(matrices,titles)):
        ax = fig.add_subplot(1,2,i+1,projection='3d')
        ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),zlim=(-1.3,1.3))
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=24,azim=34)
        ax.set_axis_off()
        for j,label in enumerate(['x','y','z']):
            e=np.eye(3)[:,j]
            ax.quiver(0,0,0,*(1.18*e),color=GRAY,linestyle=':',linewidth=1.2,arrow_length_ratio=.12)
            ax.text(*(1.32*e),label,color=GRAY,fontsize=13)
        for j,label in enumerate(['x_b（机头）','y_b','z_b']):
            e=matrix[:,j]
            color=ORANGE if j==0 else BLUE
            ax.quiver(0,0,0,*(.85*e),color=color,linewidth=2.4,arrow_length_ratio=.15)
            ax.text(*(.98*e),label,color=color,fontsize=12)
        ax.set_title(title,fontsize=13,pad=3)
        ax.text2D(.5,.02,('Rz(90°) Rx(90°)：机头沿 +y' if i==0 else
                         'Rx(90°) Rz(90°)：机头沿 +z'),transform=ax.transAxes,
                  ha='center',fontsize=12)
    fig.subplots_adjust(wspace=.02,top=.9,bottom=.1)
    save(fig,'rotation-order')


def moment():
    fig, axes = plt.subplots(1,2,figsize=(10.2,4.8))
    for i,ax in enumerate(axes):
        ax.set(xlim=(-.45,.95),ylim=(-.55,.75),aspect='equal')
        ax.axis('off')
        arrow(ax,(-.32,0),(.85,0),GRAY)
        arrow(ax,(0,.58),(0,-.4),GRAY)
        ax.text(.87,-.04,'+y_b',color=GRAY)
        ax.text(.03,-.42,'+z_b',color=GRAY)
        ax.plot(0,0,'o',color=INK,ms=6)
        ax.text(-.05,-.1,'质心',ha='right')
        ax.set_title(['(a) 在右翼位置施加向上的力','(b) 移到质心：同一力 + 力矩'][i],fontsize=13)
    ax=axes[0]
    ax.plot([0,.5],[0,0],color=BLUE,lw=3)
    ax.plot(.5,0,'o',color=BLUE)
    arrow(ax,(.5,0),(.5,.5),ORANGE)
    ax.text(.54,.4,'Fz = −20 N',color=ORANGE)
    ax.text(.25,-.12,'杆臂 0.5 m',color=BLUE,ha='center')
    ax=axes[1]
    arrow(ax,(0,0),(0,.5),ORANGE)
    ax.text(.05,.48,'Fz = −20 N',color=ORANGE)
    ax.add_patch(Circle((.48,.18),.055,fill=False,color=BLUE,lw=2))
    ax.plot(.48,.18,'o',color=BLUE,ms=4)
    ax.text(.48,-.18,'Mx = −10 N·m\n沿 −x_b，朝向读者',ha='center',color=BLUE,fontsize=12)
    fig.text(.5,.035,'从机尾向机头看：+x_b 指入纸面；图中长度只标示杆臂，力箭头长度为示意。',ha='center',fontsize=12)
    fig.subplots_adjust(wspace=.3,bottom=.13,top=.86)
    save(fig,'offset-force')


if __name__ == '__main__':
    flow()
    yaw()
    rotation_order()
    moment()
    destination=ROOT / 'public/figures/sources/generate-flight-foundations-figures.py'
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(__file__,destination)
    print('Generated 4 figures (SVG + PNG) and public source.')
