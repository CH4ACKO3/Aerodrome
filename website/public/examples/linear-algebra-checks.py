import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

H = jnp.array([[1., 0.], [0., 1.], [1., 1.]])
y = jnp.array([1., 2., 2.8])
W = jnp.diag(jnp.array([4., 1., 2.]))
x = jnp.array([0.8, 1.9])

def loss(x):
    r = H @ x - y
    return 0.5 * r @ W @ r

value, grad = jax.value_and_grad(loss)(x)
hessian = jax.hessian(loss)(x)
np.testing.assert_allclose(value, 0.095, atol=1e-12)
np.testing.assert_allclose(grad, [-1., -0.3], atol=1e-12)
np.testing.assert_allclose(hessian, H.T @ W @ H, atol=1e-12)

# 对角权重的白化最小二乘，不显式形成正规方程。
sqrt_w = jnp.sqrt(jnp.diag(W))
x_hat = jnp.linalg.lstsq(sqrt_w[:, None] * H, sqrt_w * y, rcond=None)[0]
np.testing.assert_allclose(x_hat, [34./35., 66./35.], atol=1e-12)

# 平面位置映射到距离和方位角；避开原点及角度分支切线。
def sensor(p):
    return jnp.array([jnp.linalg.norm(p), jnp.arctan2(p[1], p[0])])

p = jnp.array([3., 4.])
J = jax.jacfwd(sensor)(p)
np.testing.assert_allclose(J, [[0.6, 0.8], [-0.16, 0.12]], atol=1e-12)
v = jnp.array([0.2, -0.1])
_, jv = jax.jvp(sensor, (p,), (v,))
_, pullback = jax.vjp(sensor, p)
u = jnp.array([1., 0.5])
(jtu,) = pullback(u)
np.testing.assert_allclose(jv, J @ v, atol=1e-12)
np.testing.assert_allclose(jtu, J.T @ u, atol=1e-12)

# 中心差分只检查一个方向，无需构造全部差分梯度。
direction = jnp.array([0.6, -0.8])
eps = 1e-5
fd = (loss(x + eps * direction) - loss(x - eps * direction)) / (2 * eps)
np.testing.assert_allclose(fd, grad @ direction, rtol=1e-8, atol=1e-9)

# 固定状态下，对矩阵控制增益求导。
e = jnp.array([0.4, -0.2])
K = jnp.array([[2., -1.], [0.5, 3.]])
R = jnp.diag(jnp.array([2., 1.]))
def effort(K):
    control = -K @ e
    return 0.5 * control @ R @ control

np.testing.assert_allclose(jax.grad(effort)(K), R @ K @ jnp.outer(e, e), atol=1e-12)
print("通过：目标、梯度、海森矩阵、最小二乘、雅可比、JVP、VJP、方向差分、矩阵梯度")
