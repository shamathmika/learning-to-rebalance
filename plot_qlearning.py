import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from baselines import evaluate, greedy, no_rebalance
from bike_rebalance_env import BikeRebalanceEnv
from plot_results import BLUES, COLORS, FIGURES, INK, MUTED, SURFACE, daily_reward, style, summary
from train_qlearning import RESULTS, run_name

SEEDS = [0, 1, 2]
VARIANTS = {
    "Q-learning": (False, False, False),
    "Q-learning + weekend balancing": (True, False, False),
    "Q-learning + fine time": (False, True, False),
    "Q-learning + alpha decay": (False, False, True),
    "Q-learning + weekend balancing + fine time": (True, True, False),
    "Q-learning + weekend balancing + alpha decay": (True, False, True),
}
# lowest lost trips on training days, chosen before any variant was evaluated on test days
SELECTED = "Q-learning + weekend balancing + fine time"
COLORS = {**COLORS, "Q-learning": "#7a4fc9", SELECTED: "#00a3b8"}


def load_runs():
    runs = {}
    for label, flags in VARIANTS.items():
        names = [run_name(*flags, s) for s in SEEDS]
        tests = [np.load(RESULTS / f"{n}_test.npz") for n in names]
        runs[label] = {
            "rows": [t["rows"] for t in tests],
            "heat": np.mean([t["heat"] for t in tests], axis = 0),
            "rewards": [np.load(RESULTS / f"{n}.npz")["rewards"] for n in names],
        }
    return runs


def learning_curve(run, baselines_train):
    # weekend-balanced runs train on an easier day mix, so only the default variant is comparable to the baselines here
    fig, ax = plt.subplots(figsize = (8, 4.5), facecolor = SURFACE)
    style(ax)
    for name, value in baselines_train.items():
        ax.axhline(value, color = COLORS[name], linewidth = 2, linestyle = "--", label = name)
    curves = [pd.Series(r).rolling(2000, min_periods = 500).mean().to_numpy() for r in run["rewards"]]
    steps = np.arange(1, len(curves[0]) + 1) * BikeRebalanceEnv.STEPS_PER_DAY
    for i, curve in enumerate(curves):
        ax.plot(steps, curve, color = COLORS["Q-learning"], linewidth = 1, alpha = 0.4,
                label = "Individual seeds" if i == 0 else None)
    ax.plot(steps, np.mean(curves, axis = 0), color = COLORS["Q-learning"], linewidth = 2, label = "Q-learning mean")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.0f}M"))
    ax.set_xlabel("Training timesteps", color = MUTED)
    ax.set_ylabel("Episode reward (2,000-episode mean)", color = MUTED)
    ax.set_title(f"Tabular Q-learning on training days ({len(SEEDS)} seeds)", color = INK, loc = "left")
    ax.legend(frameon = False, labelcolor = INK, fontsize = 8, loc = "lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "qlearning_learning_curve.png", dpi = 200)


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
    ax.set_title("Q-learning lost trips on held-out test days", color = INK, loc = "left")
    ax.legend(frameon = False, labelcolor = INK, fontsize = 8)
    fig.tight_layout()
    fig.savefig(FIGURES / "qlearning_test_lost_trips.png", dpi = 200)


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
    fig.suptitle("Where and when trips are lost under Q-learning (weekday test days)", color = INK, x = 0.01, ha = "left")
    fig.savefig(FIGURES / "qlearning_lost_trips_heatmap.png", dpi = 200, bbox_inches = "tight")


def main():
    FIGURES.mkdir(exist_ok = True)
    runs = load_runs()
    baselines = {"No rebalancing": no_rebalance, "Greedy": greedy}
    base_test = {n: evaluate(p, return_heat = True) for n, p in baselines.items()}
    baselines_train = {n: daily_reward(evaluate(p, split = "train")) for n, p in baselines.items()}

    table = {n: summary([rows]) for n, (rows, _) in base_test.items()}
    table.update({n: summary(r["rows"]) for n, r in runs.items()})

    learning_curve(runs["Q-learning"], baselines_train)
    lost_trips_bars({n: table[n] for n in [*baselines, "Q-learning", SELECTED]})

    env = BikeRebalanceEnv(split = "test")
    heatmaps({"No rebalancing": base_test["No rebalancing"][1], "Greedy": base_test["Greedy"][1],
              SELECTED: runs[SELECTED]["heat"]}, env.station_ids, env.weekdays)

    fmt = lambda s: f"{s[0]:.1f} ± {s[1]:.1f}" if s[1] else f"{s[0]:.1f}"
    print("| Policy | Lost/day | Lost % | Weekday lost/day | Weekend lost/day | Van km/day |")
    print("|---|---|---|---|---|---|")
    for n, s in table.items():
        print(f"| {n} | {fmt(s['lost'])} | {s['pct'][0]:.2f}% | {fmt(s['weekday'])} | {fmt(s['weekend'])} | {fmt(s['km'])} |")


if __name__ == "__main__":
    main()
