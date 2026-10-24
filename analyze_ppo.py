#Step-level behavior of the saved PPO model vs. no_rebalance on the test days

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, str(Path.cwd()))
from baselines import no_rebalance
from bike_rebalance_env import BikeRebalanceEnv

model = PPO.load("models/ppo_preliminary", device = "cpu")
ppo = lambda o, e: model.predict(o, deterministic = True)[0]
env = BikeRebalanceEnv(split = "test")
names = pd.read_csv("data/processed/stations.csv").name.head(env.N_STATIONS).to_numpy()


def rollout(policy):
    rows = []
    for d in range(len(env.dates)):
        obs, _ = env.reset(options = {"day": d})
        done, t = False, 0
        while not done:
            a = policy(obs, env)
            obs, _, done, _, info = env.step(a)
            rows.append(dict(day = d, t = t, weekday = bool(env.weekdays[d]), station = int(a[0]),
                             requested = abs(int(a[1]) - env.MAX_TRANSFER), moved = info["transferred"],
                             lost_empty = info["lost_empty"], lost_full = info["lost_full"]))
            t += 1
    return pd.DataFrame(rows)


n = len(env.dates)
base, p = rollout(no_rebalance), rollout(ppo)
assert len(p) == n * env.STEPS_PER_DAY

print(f"model: models/ppo_preliminary.zip, {n} test days, deterministic policy\n")

print("1. Where the gain comes from (lost trips/day, weekday days only)")
wk = lambda df: df[df.weekday]
nwk = p.weekday.sum() / env.STEPS_PER_DAY
rush = lambda df: df[(df.t >= 12) & (df.t < 20)]  # steps 12-19 --> 6:00-10:00
for label, f in [("all hours", lambda df: df), ("6-10 a.m.", rush)]:
    for col in ["lost_empty", "lost_full"]:
        b, q = f(wk(base))[col].sum() / nwk, f(wk(p))[col].sum() / nwk
        print(f"   {label:9} {col:10}  no_rebalance {b:5.1f}  ppo {q:5.1f}  diff {b - q:+5.1f}")

print("\n2. Where and when PPO moves the van (all test days)")
share = p.station.value_counts(normalize = True)
for s, f in share.head(5).items():
    print(f"   {100 * f:5.1f}% of steps at station {s:2d} ({names[s]})")
print(f"   top 4 stations: {100 * share.head(4).sum():.1f}% of steps")
print(f"   bikes moved/day {p.moved.sum() / n:.1f}, of which before 6 a.m. {100 * p[p.t < 12].moved.sum() / p.moved.sum():.1f}%")

print("\n3. Requested vs. executed transfers (all test days)")
print(f"   requested {p.requested.sum() / n:.1f} bikes/day, executed {p.moved.sum() / n:.1f}"
      f" -> {100 * p.moved.sum() / p.requested.sum():.1f}% executed")
