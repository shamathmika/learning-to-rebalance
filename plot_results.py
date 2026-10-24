import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from baselines import evaluate, greedy, no_rebalance
from train_dqn import RESULTS, dqn_policy

FIGURES = Path(__file__).parent / "figures"
COLORS = {"No rebalancing": "#2a78d6", "Greedy": "#eb6834", "DQN": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors = MUTED)
    ax.yaxis.grid(True, color = GRID, linewidth = 0.8)
    ax.set_axisbelow(True)


def daily_reward(rows):
    return -(rows[:, 1] + 0.1 * rows[:, 3]).mean()


def learning_curve(monitor, baselines_train):
    m = pd.read_csv(monitor, skiprows = 1)
    steps = m.l.cumsum()
    smooth = m.r.rolling(200, min_periods = 50).mean()

    fig, ax = plt.subplots(figsize = (7, 4), facecolor = SURFACE)
    style(ax)
    ax.plot(steps, smooth, color = COLORS["DQN"], linewidth = 2)
    ax.annotate("DQN", (steps.iloc[-1], smooth.iloc[-1]), xytext = (6, 0), textcoords = "offset points",
                va = "center", color = INK)
    for name, value in baselines_train.items():
        ax.axhline(value, color = COLORS[name], linewidth = 2, linestyle = "--")
        ax.annotate(name, (0, value), xytext = (4, 4), textcoords = "offset points", color = INK)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("Training timesteps", color = MUTED)
    ax.set_ylabel("Episode reward (200-episode mean)", color = MUTED)
    ax.set_title("DQN learning curve on training days", color = INK, loc = "left")
    fig.tight_layout()
    fig.savefig(FIGURES / "dqn_learning_curve.png", dpi = 200)


def lost_trips_bars(results):
    groups = ["Weekday", "Weekend"]
    x = np.arange(len(groups))
    width = 0.26
    fig, ax = plt.subplots(figsize = (7, 4), facecolor = SURFACE)
    style(ax)
    for i, (name, rows) in enumerate(results.items()):
        values = [rows[rows[:, 0] == 1, 1].mean(), rows[rows[:, 0] == 0, 1].mean()]
        bars = ax.bar(x + (i - 1) * width, values, width - 0.02, color = COLORS[name], label = name)
        ax.bar_label(bars, fmt = "%.1f", padding = 3, color = INK, fontsize = 9)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Lost trips per day", color = MUTED)
    ax.set_title("Lost trips on held-out test days", color = INK, loc = "left")
    ax.legend(frameon = False, labelcolor = INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "test_lost_trips.png", dpi = 200)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type = int, default = 0)
    args = p.parse_args()

    FIGURES.mkdir(exist_ok = True)
    name = f"dqn_seed{args.seed}"
    model = DQN.load(RESULTS / name)
    policies = {"No rebalancing": no_rebalance, "Greedy": greedy, "DQN": dqn_policy(model)}

    results = {n: evaluate(pol) for n, pol in policies.items()}
    baselines_train = {n: daily_reward(evaluate(policies[n], split = "train")) for n in ["No rebalancing", "Greedy"]}

    learning_curve(RESULTS / f"{name}.monitor.csv", baselines_train)
    lost_trips_bars(results)

    print("| Policy | Lost/day | Lost % | Weekday lost/day | Weekend lost/day | Van km/day |")
    print("|---|---|---|---|---|---|")
    for n, rows in results.items():
        wd, we = rows[:, 0] == 1, rows[:, 0] == 0
        print(f"| {n} | {rows[:, 1].mean():.1f} | {100 * rows[:, 1].sum() / rows[:, 2].sum():.2f}% | "
              f"{rows[wd, 1].mean():.1f} | {rows[we, 1].mean():.1f} | {rows[:, 3].mean():.1f} |")


if __name__ == "__main__":
    main()
