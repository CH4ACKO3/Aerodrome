"""第一章的共轭更新和抽样例，只依赖 NumPy。

从 ngc/ 运行：python examples/bayesian_update.py
Beta 例表示一个固定但未知的成功率；Gaussian 例区分未知位置与下一读数。
这些是教学设定，不是实际传感器校准结果。
"""
import json
from math import lgamma
import numpy as np


def beta_density(theta, alpha, beta):
    """在 (0,1) 的网格上计算 Beta 密度，使用 log 形式处理归一化常数。"""
    log_normalizer = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
    return np.exp((alpha - 1) * np.log(theta)
                  + (beta - 1) * np.log1p(-theta) - log_normalizer)


def conjugate_updates():
    alpha, beta = 2.0, 2.0
    successes, failures = 8, 2
    post_a, post_b = alpha + successes, beta + failures
    # θ 网格只服务于显示；后验参数和均值由解析更新给出。
    theta = np.linspace(0.0001, 0.9999, 1000)
    likelihood = theta ** successes * (1 - theta) ** failures
    # 相对似然不是 θ 上的密度，此处只按解析峰值 0.8 归一以便画图。
    likelihood /= .8 ** successes * .2 ** failures
    prior_mean, prior_variance = 10.0, 1.0  # m、m²
    observation, noise_variance = 12.0, .25  # m、m²
    posterior_variance = 1 / (1 / prior_variance + 1 / noise_variance)
    posterior_mean = posterior_variance * (prior_mean / prior_variance
                                           + observation / noise_variance)
    return dict(theta=theta.tolist(), prior_density=beta_density(theta, alpha, beta).tolist(),
                posterior_density=beta_density(theta, post_a, post_b).tolist(),
                relative_likelihood=likelihood.tolist(), beta_prior=[alpha, beta],
                beta_posterior=[post_a, post_b], posterior_mean=post_a/(post_a+post_b),
                mle=successes/(successes+failures), posterior_mode=(post_a-1)/(post_a+post_b-2),
                gaussian=dict(posterior_mean_m=posterior_mean,
                              posterior_variance_m2=posterior_variance,
                              predictive_variance_m2=posterior_variance+noise_variance))


def sampling_demo(seed=7):
    """同一 N(10,0.5²) 分布的嵌套样本；多样本直方图仍不是精确密度。"""
    rng = np.random.default_rng(seed)
    # ε~N(0,1)，线性变换得到读数 Y=10m+0.5m ε。
    observations = 10.0 + .5 * rng.standard_normal(3000)
    grid = np.linspace(8, 12, 501)
    density = np.exp(-.5*((grid-10)/.5)**2) / (.5*np.sqrt(2*np.pi))
    return dict(seed=seed, observations_m=observations.tolist(), grid_m=grid.tolist(),
                density_per_m=density.tolist(), small_sample_count=30,
                mean_30_m=float(observations[:30].mean()), mean_3000_m=float(observations.mean()))


if __name__ == '__main__':
    print(json.dumps(dict(conjugate=conjugate_updates(), sampling=sampling_demo()),
                     ensure_ascii=False, indent=2))
