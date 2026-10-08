"""Generate Chapter 4 numerical-integration figure from actual time stepping.

Run from website/:
uv run --project ../ngc --locked --extra control python scripts/generate-flight-numerics-figures.py
Requires the installed Aerodrome ngc package, JAX, NumPy and Matplotlib.
Uses scripts/teaching.mplstyle; --font selects another Chinese font.
"""
from pathlib import Path
import argparse
import shutil

import jax
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import font_manager
from aerodrome.core.integrators import rk4


def decay_rhs(t, x, inputs, tau):
    return -x / tau


def integrate(method, dt, duration=2.0, tau=0.5):
    """Dimensionless x, seconds for dt/duration/tau; inputs remain empty."""
    count = round(duration / dt)
    time = np.arange(count + 1) * dt
    state = [1.0]
    for t in time[:-1]:
        x = state[-1]
        k1 = decay_rhs(t, x, (), tau)
        if method == "Euler":
            following = x + dt * k1
        elif method == "Heun":
            k2 = decay_rhs(t + dt, x + dt * k1, (), tau)
            following = x + dt * (k1 + k2) / 2
        else:
            following = rk4(decay_rhs, t, x, (), tau, dt)
        state.append(float(following))
    return time, np.asarray(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", default="/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    website = Path.cwd()
    plt.style.use(website / "scripts/teaching.mplstyle")
    font_manager.fontManager.addfont(args.font)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=args.font).get_name()
    output = website / "public/figures/flight-models"
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), layout="constrained")
    methods = ("Euler", "Heun", "RK4")
    styles = (("#2563a6", "-", "o"), ("#bc5930", "--", "s"), ("#775da6", "-.", "^"))
    exact_time = np.linspace(0, 2, 401)
    axes[0].plot(exact_time, np.exp(-2 * exact_time), color="#6e7b86", linestyle=":", label="解析解")
    steps = .5 / 2 ** np.arange(5)
    for method, (color, line, marker) in zip(methods, styles, strict=True):
        time, state = integrate(method, .25)
        axes[0].plot(time, state, color=color, linestyle=line, marker=marker, label=method)
        errors = np.array([abs(integrate(method, dt)[1][-1] - np.exp(-4)) for dt in steps])
        order = np.log2(errors[-2] / errors[-1])
        axes[1].loglog(steps, errors, color=color, linestyle=line, marker=marker, label=method)
        print(f"{method}: h=.25 end={state[-1]:.10f}; errors={errors}; final observed order={order:.4f}")
    axes[0].set(title="(a) 同一步长 h = 0.25 s", xlabel="时间 t / s", ylabel="归一化状态 x / 1", xlim=(0, 2), ylim=(0, 1.03))
    axes[1].set(title="(b) 缩小步长后的终点误差", xlabel="积分步长 h / s", ylabel="|x(2 s) − exp(−4)| / 1")
    axes[1].set_xticks(steps[::-1], ["0.03125", "0.0625", "0.125", "0.25", "0.5"])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    for ax in axes:
        ax.grid(True)
        ax.legend()
    for extension in ("svg", "png"):
        figure.savefig(output / f"numerics-decay-convergence.{extension}")
    plt.close(figure)
    public_source = website / "public/figures/sources/generate-flight-numerics-figures.py"
    public_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), public_source)


if __name__ == "__main__":
    main()
