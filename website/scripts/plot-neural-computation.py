"""神经网络的前向数值与反向梯度；数值直接取自同页 NumPy 例子。"""
import argparse
from pathlib import Path
import runpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
plt.style.use(Path(__file__).with_name('teaching.mplstyle'))
font = FontProperties(fname=args.font)
d = runpy.run_path(str(root/'ngc/examples/mlp_backprop.py'))['demo']()
blue, orange, gray, purple = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(figsize=(13, 6.5), layout='constrained')
ax.set(xlim=(-.6,12.6), ylim=(-.7,5.1)); ax.axis('off')

def label(x,y,text,color=gray,size=15):
    ax.text(x,y,text.replace('ᵀ', r'$^{\mathsf{T}}$'),ha='center',va='center',fontproperties=font,fontsize=size,color=color)

def box(x,y,title,value,color):
    ax.add_patch(FancyBboxPatch((x-1.05,y-.45),2.1,.9,boxstyle='round,pad=0.04',
                               edgecolor=color,facecolor='white',linewidth=1.5))
    label(x,y+.19,title,color,14)
    label(x,y-.19,value,color,15)

def arrow(x1,x2,y,color):
    ax.annotate('',xy=(x2,y),xytext=(x1,y),arrowprops={'arrowstyle':'->','color':color,'lw':1.7})

f=d['before']; delta=f['residual']; dh=d['parameters']['W2'].T[:,0]*delta
label(6,4.65,'前向：用固定参数计算中间量、预测和损失',blue,18)
for x,title,value in [(0.6,'输入 x','[1, 2]ᵀ'),(3.3,'线性层 z',str(f['z'].round(4).tolist())+'ᵀ'),
                       (6.,'激活 h',str(f['hidden'].round(4).tolist())+'ᵀ'),
                       (8.7,'预测 ŷ',f"{f['prediction']:.2f}"),(11.4,'损失 L',f"{f['loss']:.4f}")]:
    box(x,3.6,title,value,blue)
for start,end in [(1.7,2.2),(4.4,4.9),(7.1,7.6),(9.8,10.3)]:arrow(start,end,3.6,blue)
for x,text in [(1.95,'W₁x + b₁'),(4.65,'ReLU(z)'),(7.35,'W₂h + b₂'),(10.05,'½(ŷ − y)²')]:label(x,4.28,text,blue,13)
label(6,2.48,'反向：沿相反方向应用链式法则，参数保持不变',orange,18)
for x,title,value in [(1.05,'输入梯度 ∂L/∂x',str(d['input_gradient'].round(4).tolist())+'ᵀ'),
                       (4.35,'δ₁ = ∂L/∂z',str(d['gradients']['b1'].round(4).tolist())+'ᵀ'),
                       (7.65,'∂L/∂h',str(dh.round(4).tolist())+'ᵀ'),
                       (10.95,'δ₂ = ∂L/∂ŷ',f'{delta:.2f}')]:box(x,1.4,title,value,orange)
for start,end in [(9.85,8.75),(6.55,5.45),(3.25,2.15)]:arrow(start,end,1.4,orange)
for x,text in [(9.3,'乘 W₂ᵀ'),(6.,'逐元素乘 ReLU′(z)'),(2.7,'乘 W₁ᵀ')]:label(x,.65,text,orange,13)
label(6,-.08,'参数梯度来自外积：∂L/∂W₁ = δ₁xᵀ，∂L/∂W₂ = δ₂hᵀ；偏置梯度分别为 δ₁、δ₂',gray,15)
out=root/'website/public/figures/statistical-learning'
fig.savefig(out/'mlp-computation.svg',bbox_inches='tight')
fig.savefig(out/'mlp-computation.png',bbox_inches='tight',dpi=140)
