import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from baselines import evaluate, greedy, no_rebalance
from bike_rebalance_env import BikeRebalanceEnv
from plot_results import BLUES, COLORS, FIGURES, INK, MUTED, SURFACE, daily_reward, style, summary
from train_ppo import RESULTS, run_name

SEEDS = [0, 1, 2]
VARIANTS = {
    "PPO": (False, 0.01),
    "PPO + weekend balancing": (True, 0.01),
    "PPO, no entropy bonus": (False, 0.0),
}
COLORS = {**COLORS, "PPO": "#7a5af5", "PPO + weekend balancing": "#00a3b8", "PPO, no entropy bonus": "#9e5b2a"}
DIAGNOSTICS = {
    "train/entropy_loss": "Policy entropy",
    "train/approx_kl": "Approx. KL per update",
    "train/clip_fraction": "Clip fraction",
    "train/explained_variance": "Value explained variance",
}


def load_runs():
    runs = {}
    for label, (wb, ent) in VARIANTS.items():
        names = [run_name(wb, ent, s) for s in SEEDS]
        if not all((RESULTS / f"{n}_test.npz").exists() for n in names):
            print(f"skipping {label}: missing runs")
            continue
        tests = [np.load(RESULTS / f"{n}_test.npz") for n in names]
        runs[label] = {
            "rows": [t["rows"] for t in tests],
            "heat": np.mean([t["heat"] for t in tests], axis = 0),
            "monitors": [pd.read_csv(RESULTS / f"{n}.monitor.csv", skiprows = 1) for n in names],
            "logs": [pd.read_csv(RESULTS / f"{n}_log" / "progress.csv") for n in names],
        }
    return runs


def mean_std(curves):
    return np.mean(curves, axis = 0), np.std(curves, axis = 0)


def learning_curves(runs, baselines_train):
    grid = np.linspace(20_000, 1_000_000, 200)
    fig, ax = plt.subplots(figsize = (8, 4.5), facecolor = SURFACE)
    style(ax)
    for name, value in baselines_train.items():
        ax.axhline(value, color = COLORS[name], linewidth = 2, linestyle = "--", label = name)
    for label, run in runs.items():
        # weekend-balanced runs train on an easier day mix, so their reward isn't comparable to the baselines here
        if VARIANTS[label][0]:
            continue
        mean, std = mean_std([np.interp(grid, m.l.cumsum(), m.r.rolling(200, min_periods = 50).mean())
                              for m in run["monitors"]])
        ax.plot(grid, mean, color = COLORS[label], linewidth = 2, label = label)
        ax.fill_between(grid, mean - std, mean + std, color = COLORS[label], alpha = 0.15, linewidth = 0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("Training timesteps", color = MUTED)
    ax.set_ylabel("Episode reward (200-episode mean)", color = MUTED)
    ax.set_title(f"PPO variants on training days (mean ± std, {len(SEEDS)} seeds)", color = INK, loc = "left")
    ax.legend(frameon = False, labelcolor = INK, fontsize = 8, loc = "lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "ppo_learning_curve.png", dpi = 200)


def diagnostics(runs):
    fig, axes = plt.subplots(2, 2, figsize = (10, 6), facecolor = SURFACE, sharex = True)
    for ax, (col, title) in zip(axes.flat, DIAGNOSTICS.items()):
        style(ax)
        for label, run in runs.items():
            grid = run["logs"][0]["time/total_timesteps"].to_numpy()
            values = [log[col].to_numpy()[:len(grid)] for log in run["logs"]]
            mean, std = mean_std(-np.array(values) if col == "train/entropy_loss" else values)
            ax.plot(grid, mean, color = COLORS[label], linewidth = 1.5, label = label)
            ax.fill_between(grid, mean - std, mean + std, color = COLORS[label], alpha = 0.15, linewidth = 0)
        ax.set_title(title, color = INK, loc = "left", fontsize = 10)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    axes[1, 0].set_xlabel("Training timesteps", color = MUTED)
    axes[1, 1].set_xlabel("Training timesteps", color = MUTED)
    axes[0, 0].legend(frameon = False, labelcolor = INK, fontsize = 8)
    fig.tight_layout()
    fig.savefig(FIGURES / "ppo_training_diagnostics.png", dpi = 200)


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
    ax.set_title("PPO lost trips on held-out test days", color = INK, loc = "left")
    ax.legend(frameon = False, labelcolor = INK, fontsize = 8)
    fig.tight_layout()
    fig.savefig(FIGURES / "ppo_test_lost_trips.png", dpi = 200)


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
    fig.suptitle("Where and when trips are lost under PPO (weekday test days)", color = INK, x = 0.01, ha = "left")
    fig.savefig(FIGURES / "ppo_lost_trips_heatmap.png", dpi = 200, bbox_inches = "tight")


def main():
    FIGURES.mkdir(exist_ok = True)
    runs = load_runs()
    if not runs:
        raise SystemExit("no complete PPO runs in results/, run train_ppo.py first")
    baselines = {"No rebalancing": no_rebalance, "Greedy": greedy}
    base_test = {n: evaluate(p, return_heat = True) for n, p in baselines.items()}
    baselines_train = {n: daily_reward(evaluate(p, split = "train")) for n, p in baselines.items()}

    table = {n: summary([rows]) for n, (rows, _) in base_test.items()}
    table.update({n: summary(r["rows"]) for n, r in runs.items()})

    learning_curves(runs, baselines_train)
    diagnostics(runs)
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
