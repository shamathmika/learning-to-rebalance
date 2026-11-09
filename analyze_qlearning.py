"""Prints the Q-learning numbers quoted in the report that the plots and tables don't.

Needs all 6 variants for seeds 0-1-2 in results/, each also scored on training days
with --eval-only --split train.

    python analyze_qlearning.py
"""
import numpy as np
import pandas as pd

from baselines import evaluate, greedy, no_rebalance
from bike_rebalance_env import BikeRebalanceEnv
from plot_qlearning import SEEDS, SELECTED, VARIANTS
from plot_results import daily_reward, summary
from train_qlearning import FINE_TIME_EDGES, RESULTS, TIME_EDGES, ZoneQAgent, run_name

# half-hour steps
TIME_WINDOWS = {"6-10am": (12, 20), "10am-5pm": (20, 34), "5-8pm": (34, 40)}


def training_day_scores():
    print("Variants on training days (lowest lost/day is the reported variant)")
    print("| Variant | Lost/day | Weekday | Weekend | Van km/day | Per seed |")
    print("|---|---|---|---|---|---|")
    fmt = lambda s: f"{s[0]:.1f} ± {s[1]:.1f}"
    for label, flags in VARIANTS.items():
        rows = [np.load(RESULTS / f"{run_name(*flags, s)}_train.npz")["rows"] for s in SEEDS]
        s = summary(rows)
        per_seed = ", ".join(f"{r[:, 1].mean():.1f}" for r in rows)
        print(f"| {label} | {fmt(s['lost'])} | {fmt(s['weekday'])} | {fmt(s['weekend'])} | {fmt(s['km'])} | {per_seed} |")


def learning_curve_values():
    flags = VARIANTS["Q-learning"]
    curves = [pd.Series(np.load(RESULTS / f"{run_name(*flags, s)}.npz")["rewards"]).rolling(2000, min_periods = 500).mean()
              for s in SEEDS]
    mean = np.mean(curves, axis = 0)
    steps = np.arange(1, len(mean) + 1) * BikeRebalanceEnv.STEPS_PER_DAY
    none, best = (daily_reward(evaluate(p, split = "train")) for p in (no_rebalance, greedy))
    print(f"\nLearning curve (default variant, 2,000-episode mean reward)")
    print(f"baselines: no rebalancing {none:.1f}, greedy {best:.1f}")
    print(f"mean passes no rebalancing at {steps[np.nanargmax(mean > none)]:,} steps, ends at {mean[-1]:.1f}")


def weekday_losses_by_time():
    env = BikeRebalanceEnv(split = "test")
    panels = {"No rebalancing": evaluate(no_rebalance, return_heat = True)[1],
              "Greedy": evaluate(greedy, return_heat = True)[1],
              SELECTED: np.mean([np.load(RESULTS / f"{run_name(*VARIANTS[SELECTED], s)}_test.npz")["heat"] for s in SEEDS],
                                axis = 0)}
    print("\nLost trips per weekday by time of day (test days)")
    for name, heat in panels.items():
        by_step = heat[env.weekdays].mean(axis = 0).sum(axis = 1)
        print(f"{name}: " + ", ".join(f"{w} {by_step[a:b].sum():.1f}" for w, (a, b) in TIME_WINDOWS.items()))


def seed_behavior(label):
    env = BikeRebalanceEnv(split = "train")
    balance_weekends, fine_time, alpha_decay = VARIANTS[label]
    print(f"\n{label}: behavior on weekday training days")
    for seed in SEEDS:
        saved = np.load(RESULTS / f"{run_name(balance_weekends, fine_time, alpha_decay, seed)}.npz")
        agent = ZoneQAgent(env, saved["zones"], FINE_TIME_EDGES if fine_time else TIME_EDGES)
        agent.q, agent.visits = saved["q"], saved["visits"]
        lost, moved, loads = [], [], []
        for day in np.flatnonzero(env.weekdays):
            obs, _ = env.reset(options = {"day": int(day)})
            day_lost = day_moved = 0
            done = False
            while not done:
                loads.append(obs["van_load"])
                obs, _, done, _, info = env.step(agent.policy(obs, env))
                day_lost += info["lost"]
                day_moved += info["transferred"]
            lost.append(day_lost)
            moved.append(day_moved)
        print(f"seed {seed}: {np.mean(lost):.1f} lost/day, {np.mean(moved):.0f} bikes moved/day, mean van load {np.mean(loads):.1f}")


def main():
    training_day_scores()
    learning_curve_values()
    weekday_losses_by_time()
    seed_behavior("Q-learning")
    seed_behavior("Q-learning + alpha decay")


if __name__ == "__main__":
    main()
