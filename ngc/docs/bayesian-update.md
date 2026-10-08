# 贝叶斯更新与抽样：先验、数据和新观测

对应[1.2 概率论](/Aerodrome/chapters/01-mathematical-tools/02-probability/)。在仓库 `ngc/` 目录、已安装 NumPy 的 Python 环境中运行：

```sh
python3 examples/bayesian_update.py > bayesian-update.json
```

也可以下载 [bayesian_update.py](/Aerodrome/downloads/bayesian_update.py) 单独运行。`conjugate` 字段给出先验 Beta(2,2)、8 成功 2 失败后的 Beta(10,4)，以及供配图的参数网格、先验和后验密度。`relative_likelihood` 只按最大似然点的高度归一，不是成功率上的概率密度。后验均值为 5/7，最大似然估计为 0.8，后验众数为 0.75。

`conjugate.gaussian` 对位置先验 N(10 m,1 m²)、观测12 m、已知噪声方差0.25 m²完成更新：后验均值11.6 m、方差0.2 m²，下一次独立噪声读数的预测方差为0.45 m²。程序使用解析式；密度网格用于显示，不用数值网格代替共轭计算。

`sampling` 使用 NumPy 默认随机数发生器、种子7，抽取3000个 N(10 m,0.25 m²) 读数。左图使用其中前30个，两图共用理论密度和组距。`mean_30_m`、`mean_3000_m` 是这次样本均值；增加样本量会减小重复实验中均值的波动尺度，但不保证每次绝对误差都下降。更换种子可观察另一批样本。

本程序没有训练 VAE 或 GAN。教材中的生成模型段落用于解释先验抽样、后验推断和隐式分布，不能把这里的高斯抽样结果当作生成网络的训练结果。
