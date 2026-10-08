"""为统计学习第二章生成 15 幅教学图；全部数据来自同章短例或解析公式。

从 website/ 运行：python scripts/plot-statistical-learning.py --font /path/to/font.ttf
普通函数短例在 ngc/examples/statistical_learning.py。SVG 用于网页，PNG 用于核对。
"""
import argparse
from pathlib import Path
import runpy
from math import factorial

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'public/figures/statistical-learning')
args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
plt.style.use(Path(__file__).with_name('teaching.mplstyle'))
font = FontProperties(fname=args.font)
blue, orange, gray, purple = plt.rcParams['axes.prop_cycle'].by_key()['color']
examples = runpy.run_path(str(Path(__file__).resolve().parents[2]/'ngc/examples/statistical_learning.py'))
data = {name: function() for name, function in examples['DEMOS'].items()}


def save(fig, name):
    for i, ax in enumerate(fig.axes):
        if len(fig.axes)>1: ax.set_title(f'({chr(97+i)}) {ax.get_title()}')
        labels = [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]
        if ax.get_legend(): labels += ax.get_legend().get_texts()
        for label in labels:
            size = max(12, label.get_fontsize())
            label.set_fontproperties(font); label.set_fontsize(size)
    fig.savefig(args.output/f'{name}.svg', bbox_inches='tight')
    fig.savefig(args.output/f'{name}.png', bbox_inches='tight', dpi=140)
    plt.close(fig)
    print(name)


# 两幅图使用同一生成式模型，密度与后验概率的纵轴明确区分。
d = data['lda']; fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout='constrained')
for c, color, style in [(0,blue,'-'),(1,orange,'--')]:
    axes[0].plot(d['x'], d['density'][:,c], color=color, linestyle=style, label=f'类别 {c}，均值 {2*c}')
axes[0].set(xlabel='特征 x（无量纲）', ylabel='类别条件密度', title='先描述每一类的输入'); axes[0].set_ylim(0, .56); axes[0].legend(loc='upper right')
axes[1].plot(d['x'], d['posterior_class_1'], color=blue)
axes[1].axhline(.5, color=gray, ls=':');axes[1].axvline(1, color=gray, ls=':')
axes[1].scatter([1.5],[d['probability_at_x_1_5']],color=orange)
axes[1].annotate('x = 1.5，概率 ≈ 0.731',xy=(1.5,d['probability_at_x_1_5']),xytext=(-2.5,.85),arrowprops={'arrowstyle':'->'})
axes[1].set(xlabel='特征 x（无量纲）', ylabel='P(y = 1 | x)', title='再归一化得到类别后验', ylim=(0,1.05))
save(fig,'lda')

# 左侧为解析映射；右侧来自完整训练流程的验证损失，不使用测试结果。
d=data['logistic'];fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
z=np.linspace(-6,6,300);axes[0].plot(z,examples['sigmoid'](z),color=blue)
axes[0].scatter([-2,0,2],examples['sigmoid']([-2,0,2]),color=orange)
axes[0].set(xlabel='得分 z（无量纲）',ylabel='类别 1 概率',title='同一个映射：从得分到概率')
axes[1].bar([0,1,2],d['validation_nll'],color=[blue,gray,gray],width=.6)
for i,value in enumerate(d['validation_nll']):axes[1].text(i,value+.015,f'{value:.3f}',ha='center')
axes[1].set(xticks=[0,1,2],xticklabels=['0','0.03','0.3'],xlabel='正则化强度 λ',ylabel='验证集平均负对数似然',ylim=(0,.72),title='同一划分：比较三个训练结果')
save(fig,'logistic')

d=data['regression'];fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained');x=np.linspace(-2.2,2.2,250)
ax.scatter(d['x'],d['y'],color=gray,label='训练样本',zorder=3)
ax.plot(x,np.column_stack([np.ones(len(x)),x])@d['linear_weights'],color=blue,label='特征 [1, x]')
ax.plot(x,np.column_stack([np.ones(len(x)),x,x*x])@d['quadratic_weights'],'--',color=orange,label='特征 [1, x, x²]')
ax.set(xlabel='输入 x（无量纲）',ylabel='响应 y（无量纲）',title='对参数线性的模型，也能画出弯曲函数');ax.legend()
save(fig,'regression')

d=data['glm'];fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
x=np.linspace(0,2,100);axes[0].plot(x,2*2**x,color=blue);axes[0].scatter(d['load'],d['mean_count_per_minute'],color=orange)
axes[0].set(xlabel='负载 x（无量纲）',ylabel='一分钟内平均计数',title='对数均值随负载线性变化')
k=np.arange(11);mass=np.array([np.exp(-2)*2.**n/factorial(n) for n in k]);axes[1].bar(k,mass,color=blue)
axes[1].set(xlabel='一分钟内实际次数',ylabel='单点概率',title='均值 2 不表示每次都发生 2 次',xticks=[0,2,4,6,8,10])
save(fig,'glm')

d=data['mlp'];fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
for ax,points,title in [(axes[0],d['inputs'],'输入空间：XOR'),(axes[1],d['hidden'],'隐藏空间：保留输入差异')]:
    for label,color,marker in [(0,blue,'o'),(1,orange,'s')]:
        group=points[d['labels']==label];ax.scatter(*group.T,color=color,marker=marker,s=100,label=f'类别 {label}',zorder=3)
    ax.set(xlim=(-.2,1.45),ylim=(-.2,1.55),xticks=[0,1],yticks=[0,1],title=title);ax.set_aspect('equal');ax.legend(loc='upper right')
axes[0].set(xlabel='输入 x₁',ylabel='输入 x₂')
axes[1].plot([-.15,.65],[.65,-.15],'--',color=gray);axes[1].text(.12,.12,'两点重合',color=blue)
axes[1].set(xlabel='ReLU(x₁ − x₂)',ylabel='ReLU(x₂ − x₁)')
save(fig,'mlp')

# 图内写数值，不让灰度深浅独自承担含义；各面板按自己的运算量标注。
d=data['convolution'];fig,axes=plt.subplots(1,3,figsize=(11,4),layout='constrained')
for ax,array,title in zip(axes,[d['image'],d['kernel'],d['valid_response']],['输入 5 × 5','共享模板 3 × 2','响应 3 × 4']):
    ax.imshow(array,cmap='Blues',vmin=min(0,array.min()),vmax=max(1,array.max()),interpolation='nearest')
    for (r,c),value in np.ndenumerate(array):ax.text(c,r,f'{value:g}',ha='center',va='center',color='white' if value>array.min()+.65*(array.max()-array.min()) else '#192b35')
    ax.set(xticks=[],yticks=[],title=title)
save(fig,'convolution')

d=data['sequence'];fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
axes[0].imshow(d['causal_weights'],cmap='Blues',vmin=0,vmax=1)
for (r,c),value in np.ndenumerate(d['causal_weights']):axes[0].text(c,r,f'{value:.3f}',ha='center',va='center',color='white' if value>.65 else '#192b35')
axes[0].set(xticks=[0,1,2],xticklabels=['1','2','3'],yticks=[0,1,2],yticklabels=['1','2','3'],xlabel='被读取的位置',ylabel='当前查询位置',title='每行已知位置权重之和为 1')
axes[1].plot([1,2,3],d['outputs'],'o-',color=blue,label='注意力输出')
axes[1].plot([1,2,3],d['values'],'s--',color=orange,label='当前位置的原始值')
axes[1].set(xlabel='当前查询位置',ylabel='无量纲数值',xticks=[1,2,3],ylim=(0,35),title='输出汇总了当前可见的信息');axes[1].legend()
save(fig,'sequence')

d=data['neighbors'];fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained')
for k,color,style in [('1',blue,'-'),('3',orange,'--')]:ax.plot(d['query'],d['predictions'][k],color=color,linestyle=style,label=f'K = {k}')
ax.scatter(d['x'],d['y'],color=gray,s=60,zorder=3,label='训练样本')
ax.set(xlabel='查询 x（无量纲）',ylabel='预测或观测 y（无量纲）',title='改变邻居数量，就是改变局部平均范围');ax.legend()
save(fig,'neighbors')

d=data['kernels'];fig,ax=plt.subplots(figsize=(8.5,4.3),layout='constrained')
ax.fill_between(d['query'],d['mean']-2*d['latent_sd'],d['mean']+2*d['latent_sd'],color=blue,alpha=.14,label='潜在函数均值 ±2 SD')
ax.plot(d['query'],d['mean'],color=blue,label='后验均值');ax.scatter(d['x'],d['y'],color=orange,zorder=3,label='含噪观测')
ax.axhline(0,color=gray,ls=':',lw=1)
ax.set(xlabel='输入 x（无量纲）',ylabel='潜在响应 f（无量纲）',title='固定核和噪声设定下的 GP 条件后验');ax.legend(loc='upper left')
save(fig,'kernels')

d=data['trees'];fig,ax=plt.subplots(figsize=(8.5,4.3),layout='constrained');query=np.linspace(-.2,3.2,300)
first=np.where(query<=d['threshold'],d['stump_prediction'][0],d['stump_prediction'][-1])
correction=(d['boosted_prediction']-d['stump_prediction'])/.5
boosted=first+.5*np.where(query<=d['residual_threshold'],correction[0],correction[-1])
ax.plot(query,first,color=blue,label=f'第一棵树：训练 SSE = {d["stump_sse"]:.3f}')
ax.plot(query,boosted,'--',color=orange,label=f'加一次修正：SSE = {d["boosted_sse"]:.3f}')
ax.scatter(d['x'],d['y'],color=gray,zorder=3,label='训练样本')
ax.set(xlabel='输入 x（无量纲）',ylabel='预测或观测 y（无量纲）',title='平方损失下，让下一棵树拟合当前残差');ax.legend()
save(fig,'trees')

d=data['augmentation'];fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
for ax,before,after,title,unit in [(axes[0],d['position'],d['rotated_position'],'输入位置也旋转','m'),(axes[1],d['velocity_label'],d['rotated_velocity_label'],'向量标签必须同步旋转','m/s')]:
    for vector,color,label in [(before,blue,'变换前'),(after,orange,'变换后')]:
        ax.annotate('',xy=vector,xytext=(0,0),arrowprops={'arrowstyle':'->','lw':3,'color':color})
        ax.plot([],[],color=color,label=label)
    ax.set(xlim=(-.3,2.6),ylim=(-.3,2.6),xlabel=f'横向分量（{unit}）',ylabel=f'纵向分量（{unit}）',title=title);ax.set_aspect('equal');ax.grid();ax.legend(loc='upper right')
save(fig,'augmentation')

d=data['pca'];fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
ax.plot([-2.4,2.4],np.array([-2.4,2.4])*d['direction'][1]/d['direction'][0],color=blue,label='保留的主方向')
ax.scatter(*d['x'].T,color=orange,s=65,zorder=3,label='原始样本');ax.scatter(*d['reconstructed'].T,color=blue,marker='x',s=65,zorder=3,label='一维编码后的重建')
for x,reconstruction in zip(d['x'],d['reconstructed']):ax.plot([x[0],reconstruction[0]],[x[1],reconstruction[1]],'--',color=gray)
ax.set(xlabel='特征 1（无量纲）',ylabel='特征 2（无量纲）',ylim=(-1.6,2.1),title=f'保留方差比例 {100*d["explained_variance_ratio"]:.2f}%，仍有信息被丢弃');ax.set_aspect('equal');ax.legend(loc='upper left')
save(fig,'pca')

d=data['clustering'];fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
for ax,centers,title in [(axes[0],[0.,4.],'分配：按初始中心找最近组'),(axes[1],d['history'][0]['centers'],'更新：中心移到组内均值')]:
    for group,color,marker in [(0,blue,'o'),(1,orange,'s')]:
        points=d['x'][d['assignment']==group]
        ax.scatter(points,np.zeros(len(points)),color=color,marker=marker,s=80)
        ax.scatter([centers[group]],[.23],color=color,marker='D',s=90)
        ax.text(centers[group],.36,f'中心 {centers[group]:g}',ha='center',color=color)
    ax.set(xlim=(-.7,5.7),ylim=(-.25,.75),yticks=[],xlabel='唯一特征 x（无量纲）',title=title)
save(fig,'clustering')

d=data['recommendation'];fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
for ax,array,title,hide in [(axes[0],d['observed_values'],'只给训练程序八个已知数',True),(axes[1],d['prediction'],'秩一模型预测缺失项',False)]:
    shown=np.ma.array(array,mask=~d['observed_mask'] if hide else np.zeros_like(array,dtype=bool))
    ax.imshow(shown,cmap='Blues',vmin=0,vmax=18)
    for (r,c),value in np.ndenumerate(array):
        text='?' if hide and not d['observed_mask'][r,c] else f'{value:.0f}'
        ax.text(c,r,text,ha='center',va='center',color='white' if value>9 else '#192b35')
    ax.set(xticks=[0,1,2],xticklabels=['1','2','3'],yticks=[0,1,2],yticklabels=['1','2','3'],xlabel='任务编号',ylabel='配置编号',title=title)
save(fig,'recommendation')

d=data['graphs'];fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
ax=axes[0];ax.plot([0,1,2],[0,0,0],color=gray,zorder=1)
for i,(before,after) in enumerate(zip(d['features'],d['one_round'])):
    ax.scatter([i],[0],color=[blue,orange,purple][i],s=350,zorder=2)
    ax.text(i,.24,f'节点 {i}',ha='center');ax.text(i,-.3,f'{before:g} → {after:g}',ha='center')
ax.text(1,.7,'把自身与邻居一起平均',ha='center')
ax.set(xlim=(-.5,2.5),ylim=(-.7,1),xticks=[],yticks=[],title='一轮后的特征');ax.axis('off')
for i,color,style in [(0,blue,'-'),(1,orange,'--'),(2,purple,':')]:axes[1].plot(range(9),d['history'][:,i],linestyle=style,color=color,marker=['o','s','^'][i],label=f'节点 {i}')
axes[1].set(xlabel='聚合轮数',ylabel='节点特征（无量纲）',title='多次平均后，差异逐渐缩小');axes[1].legend()
save(fig,'graphs')
