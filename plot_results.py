from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from baselines import evaluate, greedy, no_rebalance
from bike_rebalance_env import BikeRebalanceEnv
from train_dqn import RESULTS, run_name

FIGURES = Path(__file__).parent / "figures"
SEEDS = [0, 1, 2]
VARIANTS = {
    "DQN": (False, False),
    "Double DQN": (True, False),
    "DQN + weekend balancing": (False, True),
    "Double DQN + weekend balancing": (True, True),
}
COLORS = {
    "No rebalancing": "#2a78d6", "Greedy": "#eb6834", "DQN": "#1baf7a", "Double DQN": "#eda100",
    "DQN + weekend balancing": "#e87ba4", "Double DQN + weekend balancing": "#008300",
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


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


def load_runs():
    runs = {}
    for label, (double, wb) in VARIANTS.items():
        names = [run_name(double, wb, s) for s in SEEDS]
        tests = [np.load(RESULTS / f"{n}_test.npz") for n in names]
        runs[label] = {
            "rows": [t["rows"] for t in tests],
            "heat": np.mean([t["heat"] for t in tests], axis = 0),
            "monitors": [pd.read_csv(RESULTS / f"{n}.monitor.csv", skiprows = 1) for n in names],
        }
    return runs


def learning_curves(runs, baselines_train):
    grid = np.linspace(20_000, 1_000_000, 200)
    fig, ax = plt.subplots(figsize = (8, 4.5), facecolor = SURFACE)
    style(ax)
    for name, value in baselines_train.items():
        ax.axhline(value, color = COLORS[name], linewidth = 2, linestyle = "--", label = name)
    for label, run in runs.items():
        curves = [np.interp(grid, m.l.cumsum(), m.r.rolling(200, min_periods = 50).mean()) for m in run["monitors"]]
        mean, std = np.mean(curves, axis = 0), np.std(curves, axis = 0)
        ax.plot(grid, mean, color = COLORS[label], linewidth = 2, label = label)
        ax.fill_between(grid, mean - std, mean + std, color = COLORS[label], alpha = 0.15, linewidth = 0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("Training timesteps", color = MUTED)
    ax.set_ylabel("Episode reward (200-episode mean)", color = MUTED)
    ax.set_title(f"DQN variants on training days (mean ± std, {len(SEEDS)} seeds)", color = INK, loc = "left")
    ax.legend(frameon = False, labelcolor = INK, fontsize = 8, loc = "lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "dqn_learning_curve.png", dpi = 200)


def lost_trips_bars(table):
    groups = ["Weekday", "Weekend"]
    x = np.arange(len(groups))
    width = 0.8 / len(table)
    fig, ax = plt.subplots(figsize = (9, 4.5), facecolor = SURFACE)
    style(ax)
    for i, (name, r) in enumerate(table.items()):
        offset = (i - (len(table) - 1) / 2) * width
        bars = ax.bar(x + offset, [r["weekday"][0], r["weekend"][0]], width - 0.02,
                      yerr = [r["weekday"][1], r["weekend"][1]], color = COLORS[name], label = name,
                      error_kw = {"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
        ax.bar_label(bars, fmt = "%.0f", padding = 3, color = INK, fontsize = 8)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Lost trips per day", color = MUTED)
    ax.set_title("Lost trips on held-out test days", color = INK, loc = "left")
    ax.legend(frameon = False, labelcolor = INK, fontsize = 8)
    fig.tight_layout()
    fig.savefig(FIGURES / "test_lost_trips.png", dpi = 200)


def heatmaps(panels, station_ids, weekdays):
    hourly = {n: h[weekdays].mean(axis = 0).reshape(24, 2, -1).sum(axis = 1).T for n, h in panels.items()}
    vmax = max(h.max() for h in hourly.values())
    fig, axes = plt.subplots(1, len(hourly), figsize = (5 * len(hourly), 5.5), facecolor = SURFACE, sharey = True)
    for ax, (name, h) in zip(axes, hourly.items()):
        im = ax.imshow(h, aspect = "auto", cmap = BLUES, vmin = 0, vmax = vmax)
        ax.set_title(f"{name}: {h.sum():.0f} lost/weekday", color = INK, loc = "left", fontsize = 10)
        ax.set_xticks(range(0, 24, 3), [f"{t:02d}" for t in range(0, 24, 3)])
        ax.set_xlabel("Hour of day", color = MUTED)
        ax.tick_params(colors = MUTED, labelsize = 8)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_yticks(range(len(station_ids)), station_ids)
    fig.colorbar(im, ax = axes, label = "Lost trips per weekday", shrink = 0.8)
    fig.suptitle("Where and when trips are lost (weekday test days)", color = INK, x = 0.01, ha = "left")
    fig.savefig(FIGURES / "lost_trips_heatmap.png", dpi = 200, bbox_inches = "tight")


def summary(rows_list):
    stats = lambda f: (np.mean([f(r) for r in rows_list]), np.std([f(r) for r in rows_list]))
    return {
        "lost": stats(lambda r: r[:, 1].mean()),
        "pct": stats(lambda r: 100 * r[:, 1].sum() / r[:, 2].sum()),
        "weekday": stats(lambda r: r[r[:, 0] == 1, 1].mean()),
        "weekend": stats(lambda r: r[r[:, 0] == 0, 1].mean()),
        "km": stats(lambda r: r[:, 3].mean()),
    }


def main():
    FIGURES.mkdir(exist_ok = True)
    runs = load_runs()
    baselines = {"No rebalancing": no_rebalance, "Greedy": greedy}
    base_test = {n: evaluate(p, return_heat = True) for n, p in baselines.items()}
    baselines_train = {n: daily_reward(evaluate(p, split = "train")) for n, p in baselines.items()}

    table = {n: summary([rows]) for n, (rows, _) in base_test.items()}
    table.update({n: summary(r["rows"]) for n, r in runs.items()})

    learning_curves(runs, baselines_train)
    lost_trips_bars(table)

    best = min(runs, key = lambda n: table[n]["lost"][0])
    env = BikeRebalanceEnv(split = "test")
    heatmaps({"No rebalancing": base_test["No rebalancing"][1], "Greedy": base_test["Greedy"][1],
              best: runs[best]["heat"]}, env.station_ids, env.weekdays)

    fmt = lambda s: f"{s[0]:.1f} ± {s[1]:.1f}" if s[1] else f"{s[0]:.1f}"
    print("| Policy | Lost/day | Lost % | Weekday lost/day | Weekend lost/day | Van km/day |")
    print("|---|---|---|---|---|---|")
    for n, s in table.items():
        print(f"| {n} | {fmt(s['lost'])} | {s['pct'][0]:.2f}% | {fmt(s['weekday'])} | {fmt(s['weekend'])} | {fmt(s['km'])} |")


if __name__ == "__main__":
    main()
