"""三站点、两次动作的有限时域动态规划；只使用 Python 标准库。

从 ngc/ 运行：python examples/finite_horizon_dp.py
随机转移例：python examples/finite_horizon_dp.py --success-probability 0.8

状态 0、1、2 是站点编号，不是米；所有代价使用同一人为设定单位。
模型已知，执行时状态也已知。前进无论成功与否都付出代价 1。
程序计算各分支的精确期望，不以随机抽样的轨迹代替期望计算。
"""

import argparse
from math import isclose


STATES = range(3)
HORIZON = 2


def action_cost(state, action, next_values, success_probability):
    """本步代价加上下步值；停留确定发生，前进可能留在原地。"""
    if action == 0:
        return next_values[state]
    return (
        1.0
        + success_probability * next_values[state + 1]
        + (1.0 - success_probability) * next_values[state]
    )


def backward_induction(success_probability):
    """保存三个时间层的值，并保留每个状态下全部并列最优动作。"""
    values = [[0.0] * 3 for _ in range(HORIZON + 1)]
    optimal_actions = [[[] for _ in STATES] for _ in range(HORIZON)]
    values[HORIZON] = [4.0 * (2 - state) ** 2 for state in STATES]

    # 第 k 层只依赖已求出的第 k+1 层，因此从终点向更早时刻倒推。
    for k in reversed(range(HORIZON)):
        for state in STATES:
            actions = (0,) if state == 2 else (0, 1)
            candidates = [
                (action, action_cost(state, action, values[k + 1], success_probability))
                for action in actions
            ]
            values[k][state] = min(cost for _, cost in candidates)
            # 浮点计算有舍入误差；比较时保留数值上相同的最优值。
            optimal_actions[k][state] = [
                action
                for action, cost in candidates
                if isclose(cost, values[k][state], rel_tol=1e-12, abs_tol=1e-12)
            ]
    return values, optimal_actions


def evaluate_policy(policy, success_probability):
    """评价给定策略：仍从后向前计算，但不再对动作取最小值。"""
    values = [[0.0] * 3 for _ in range(HORIZON + 1)]
    values[HORIZON] = [4.0 * (2 - state) ** 2 for state in STATES]
    for k in reversed(range(HORIZON)):
        for state in STATES:
            values[k][state] = action_cost(
                state, policy[k][state], values[k + 1], success_probability
            )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--success-probability", type=float, default=1.0)
    args = parser.parse_args()
    probability = args.success_probability
    if not 0.0 <= probability <= 1.0:
        parser.error("前进成功率应位于 [0, 1]。")

    values, optimal_actions = backward_induction(probability)
    print(f"前进成功率 = {probability:g}；状态顺序 = [0, 1, 2]")
    for k in reversed(range(HORIZON + 1)):
        print(f"V_{k} = [{', '.join(f'{value:g}' for value in values[k])}]")
    print("\nk  s  停留代价  前进代价  最优动作（0=停留，1=前进）")
    for k in reversed(range(HORIZON)):
        for state in STATES:
            stay = action_cost(state, 0, values[k + 1], probability)
            advance = (
                f"{action_cost(state, 1, values[k + 1], probability):g}"
                if state < 2
                else "不允许"
            )
            print(f"{k}  {state}  {stay:g}  {advance}  {optimal_actions[k][state]}")

    # 第一次停留，最后一次有空间时前进。它是一个待评价的完整策略。
    delayed_policy = [[0, 0, 0], [1, 1, 0]]
    delayed_values = evaluate_policy(delayed_policy, probability)
    print("\n先等待策略的 V_0 = [" + ", ".join(f"{v:g}" for v in delayed_values[0]) + "]")
    # 并列时任选一个即得可执行策略；此处用列表中的第一个动作。
    selected_policy = [
        [actions[0] for actions in time_slice] for time_slice in optimal_actions
    ]
    checked_values = evaluate_policy(selected_policy, probability)
    print("最优策略单独评价的 V_0 = [" + ", ".join(f"{v:g}" for v in checked_values[0]) + "]")


if __name__ == "__main__":
    main()
