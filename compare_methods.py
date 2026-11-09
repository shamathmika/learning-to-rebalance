"""Best variant of each method on the held-out test days, with paired confidence intervals.

Needs the {run}_test.npz files of DQN and PPO in results/ as well as Q-learning's.
evaluate() replays the days in the same order, so results line up day by day across policies.
Each method is averaged over its seeds first, so the intervals reflect day-to-day variation, not seed spread.

    python compare_methods.py
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import evaluate, greedy, no_rebalance
from plot_results import FIGURES, INK, MUTED, SURFACE, style, summary
from train_dqn import RESULTS

SEEDS = [0, 1, 2]
BOOTSTRAPS = 10_000
METHODS = {
    "Q-learning + weekend balancing + fine time": "qlearning_wb_fine_seed{}",
    "PPO": "ppo_ent0.01_seed{}",
    "DQN + weekend balancing": "dqn_wb_seed{}",
}
COLORS = {
    "No rebalancing": "#2a78d6", "Greedy": "#eb6834", "Q-learning + weekend balancing + fine time": "#00a3b8",
    "PPO": "#7a5af5", "DQN + weekend balancing": "#e87ba4",
}


def load_methods():
    files = {label: [RESULTS / f"{pattern.format(s)}_test.npz" for s in SEEDS] for label, pattern in METHODS.items()}
    missing = [f.name for paths in files.values() for f in paths if not f.exists()]
    if missing:
        raise SystemExit(f"missing from results/: {', '.join(missing)}")
    return {label: [np.load(f)["rows"] for f in paths] for label, paths in files.items()}


def paired_ci(a, b, rng):
    """Mean of a - b per day, with a 95% bootstrap interval over days."""
    diff = a - b
    means = diff[rng.integers(len(diff), size = (BOOTSTRAPS, len(diff)))].mean(axis = 1)
    return diff.mean(), *np.percentile(means, [2.5, 97.5])


def comparison_table(rows, rng):
    daily = {n: np.mean([r[:, 1] for r in rs], axis = 0) for n, rs in rows.items()}
    weekday = next(iter(rows.values()))[0][:, 0] == 1
    print("| A minus B (lost/day) | All days | Weekdays | Weekends |")
    print("|---|---|---|---|")
    for a, b in itertools.combinations(reversed(list(rows)), 2):
        cells = []
        for mask in [np.ones(len(weekday), bool), weekday, ~weekday]:
            mean, low, high = paired_ci(daily[a][mask], daily[b][mask], rng)
            cells.append(f"{mean:+.1f} [{low:+.1f}, {high:+.1f}]")
        print(f"| {a} minus {b} | {' | '.join(cells)} |")


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
    ax.set_title("All methods on held-out test days", color = INK, loc = "left")
    ax.legend(frameon = False, labelcolor = INK, fontsize = 8)
    fig.tight_layout()
    fig.savefig(FIGURES / "method_comparison.png", dpi = 200)


def main():
    FIGURES.mkdir(exist_ok = True)
    rows = {"No rebalancing": [evaluate(no_rebalance)], "Greedy": [evaluate(greedy)], **load_methods()}
    table = {n: summary(rs) for n, rs in rows.items()}
    lost_trips_bars(table)

    fmt = lambda s: f"{s[0]:.1f} ± {s[1]:.1f}" if s[1] else f"{s[0]:.1f}"
    print("| Policy | Lost/day | Lost % | Weekday lost/day | Weekend lost/day | Van km/day |")
    print("|---|---|---|---|---|---|")
    for n, s in table.items():
        print(f"| {n} | {fmt(s['lost'])} | {s['pct'][0]:.2f}% | {fmt(s['weekday'])} | {fmt(s['weekend'])} | {fmt(s['km'])} |")
    print()
    comparison_table(rows, np.random.default_rng(0))


if __name__ == "__main__":
    main()
