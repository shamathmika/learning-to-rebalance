#Test-day comparison of PPO variants (mean ± std over seeds), plus checkpoints of the 3M-step runs

#   python compare_ppo.py

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from baselines import evaluate, greedy, no_rebalance
from plot_results import summary
from train_ppo import CHECKPOINT_EVERY, RESULTS, ppo_policy, run_name

SEEDS = [0, 1, 2]
VARIANTS = {
    "PPO 1M": {},
    "PPO 1M, raw obs": {"raw_obs": True},
    "PPO 3M": {"timesteps": 3_000_000},
    "PPO 3M + LR decay": {"timesteps": 3_000_000, "lr_decay": True},
    "PPO 3M + weekend time": {"timesteps": 3_000_000, "weekend_time": True},
}


def names(kwargs):
    return [run_name(False, 0.01, s, **kwargs) for s in SEEDS]


def fmt(s):
    return f"{s[0]:.1f} ± {s[1]:.1f}" if s[1] else f"{s[0]:.1f}"


def results_table(runs):
    table = {"No rebalancing": [evaluate(no_rebalance)], "Greedy": [evaluate(greedy)], **runs}
    print("| Policy | Lost/day | Lost % | Weekday lost/day | Weekend lost/day | Van km/day | Lost/day per seed |")
    print("|---|---|---|---|---|---|---|")
    for label, rows in table.items():
        s = summary(rows)
        per_seed = ", ".join(f"{r[:, 1].mean():.1f}" for r in rows) if len(rows) > 1 else ""
        print(f"| {label} | {fmt(s['lost'])} | {s['pct'][0]:.2f}% | {fmt(s['weekday'])} | {fmt(s['weekend'])}"
              f" | {fmt(s['km'])} | {per_seed} |")


def window_stats(name, end):
    # training reward and update diagnostics over the 1M steps before end
    log = pd.read_csv(RESULTS / f"{name}_log" / "progress.csv").dropna(subset = ["train/approx_kl"])
    log = log[(log["time/total_timesteps"] > end - CHECKPOINT_EVERY) & (log["time/total_timesteps"] <= end)]
    monitor = pd.read_csv(RESULTS / f"{name}.monitor.csv", skiprows = 1)
    steps = monitor.l.cumsum()
    reward = monitor.r[(steps > end - 100_000) & (steps <= end)].mean()
    return reward, log["train/approx_kl"].mean(), log["train/clip_fraction"].mean()


def convergence(label, kwargs):
    # checkpoints are saved mid-rollout, so "1M" is the policy after the last update before step 1,000,000
    print(f"\n{label}: test lost/day and training diagnostics per 1M-step window (mean ± std over seeds)")
    print("| Steps | Test lost/day | Weekday | Weekend | Train reward (last 100k) | approx_kl | clip_fraction |")
    print("|---|---|---|---|---|---|---|")
    raw, wtime = kwargs.get("raw_obs", False), kwargs.get("weekend_time", False)
    for end in range(CHECKPOINT_EVERY, kwargs["timesteps"] + 1, CHECKPOINT_EVERY):
        rows = []
        for name in names(kwargs):
            if end == kwargs["timesteps"]:
                rows.append(np.load(RESULTS / f"{name}_test.npz")["rows"])
            else:
                model = PPO.load(RESULTS / f"{name}_ckpt" / f"ppo_{end}_steps", device = "cpu")
                rows.append(evaluate(ppo_policy(model, raw, wtime)))
        s = summary(rows)
        stats = np.array([window_stats(name, end) for name in names(kwargs)])
        mean, std = stats.mean(axis = 0), stats.std(axis = 0)
        print(f"| {end / 1e6:g}M | {fmt(s['lost'])} | {fmt(s['weekday'])} | {fmt(s['weekend'])}"
              f" | {mean[0]:.1f} ± {std[0]:.1f} | {mean[1]:.4f} ± {std[1]:.4f} | {mean[2]:.3f} ± {std[2]:.3f} |")


def main():
    runs = {}
    for label, kwargs in VARIANTS.items():
        if all((RESULTS / f"{n}_test.npz").exists() for n in names(kwargs)):
            runs[label] = [np.load(RESULTS / f"{n}_test.npz")["rows"] for n in names(kwargs)]
        else:
            print(f"skipping {label}: missing runs")
    results_table(runs)
    for label, kwargs in VARIANTS.items():
        if label in runs and kwargs.get("timesteps", 0) > CHECKPOINT_EVERY:
            convergence(label, kwargs)


if __name__ == "__main__":
    main()
