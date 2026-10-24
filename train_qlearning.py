"""Tabular Q-learning with stations grouped into zones.

The full state (20 bike counts, van location and load, time step) is far too large
for a table, so it is aggregated:
    - stations are grouped into N_ZONES geographic zones (k-means on coordinates)
    - each zone is summarized by whether it has a station near empty and/or near full
    - van zone, van load (empty / partial / full), time-of-day bucket, weekday flag
Actions are "wait" or "go to zone z and pick up / drop off", carried out at the
fullest (pickup) or emptiest (drop-off) station in that zone.

    python train_qlearning.py --seed 0
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import evaluate, summarize, no_rebalance, greedy
from bike_rebalance_env import BikeRebalanceEnv, DEMAND

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent / "figures"
STATIONS = Path(__file__).parent / "data" / "processed" / "stations.csv"

N_ZONES = 3
NEAR_EMPTY, NEAR_FULL = 0.15, 0.85
# time-of-day buckets in 30-min steps: night, early, morning peak, midday, evening peak, late
TIME_EDGES = [12, 16, 20, 32, 40]  # 6am, 8am, 10am, 4pm, 8pm


def make_zones(station_ids, n_zones = N_ZONES, seed = 0):
    """K-means on station coordinates; returns a zone index per station."""
    st = pd.read_csv(STATIONS).set_index("station_id").loc[station_ids]
    xy = st[["lat", "lon"]].to_numpy()
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(50):  # several random starts, keep the tightest clustering
        centers = xy[rng.choice(len(xy), n_zones, replace = False)]
        for _ in range(100):
            zone = np.argmin(((xy[:, None] - centers[None]) ** 2).sum(-1), axis = 1)
            centers = np.array([xy[zone == z].mean(0) if (zone == z).any() else centers[z] for z in range(n_zones)])
        cost = ((xy - centers[zone]) ** 2).sum()
        if cost < best_cost and len(set(zone)) == n_zones:
            best, best_cost = zone, cost
    return best


class ZoneQAgent:
    def __init__(self, env, zones):
        self.env = env
        self.zones = zones
        self.n_zones = n = int(zones.max()) + 1
        self.members = [np.flatnonzero(zones == z) for z in range(n)]
        self.N_ACTIONS = 1 + 2 * n  # wait, then (pickup, dropoff) per zone
        self.n_states = 4 ** n * n * 3 * (len(TIME_EDGES) + 1) * 2
        self.q = np.zeros((self.n_states, self.N_ACTIONS))
        self.visits = np.zeros(self.n_states, dtype = np.int64)

    def state(self, obs):
        fill = obs["bikes_per_station"] / self.env.capacity
        s = 0
        for m in self.members:
            risk = int((fill[m] <= NEAR_EMPTY).any()) + 2 * int((fill[m] >= NEAR_FULL).any())
            s = s * 4 + risk
        load = obs["van_load"]
        load_level = 0 if load == 0 else (2 if load > self.env.VAN_CAPACITY // 2 else 1)
        time_bucket = int(np.searchsorted(TIME_EDGES, obs["time_step"], side = "right"))
        s = s * self.n_zones + int(self.zones[obs["van_location"]])
        s = s * 3 + load_level
        s = s * (len(TIME_EDGES) + 1) + time_bucket
        s = s * 2 + int(obs["weekday"])
        return s

    def to_env_action(self, a, obs):
        env = self.env
        if a == 0:
            return np.array([obs["van_location"], env.MAX_TRANSFER])  # stay, move nothing
        zone, pickup = divmod(a - 1, 2)
        pickup = pickup == 0
        m = self.members[zone]
        fill = obs["bikes_per_station"][m] / env.capacity[m]
        if pickup:
            return np.array([m[np.argmax(fill)], 2 * env.MAX_TRANSFER])
        return np.array([m[np.argmin(fill)], 0])

    def greedy_action(self, s):
        if self.visits[s] == 0:
            return 0  # unseen state: wait
        return int(np.argmax(self.q[s]))

    def policy(self, obs, env):
        return self.to_env_action(self.greedy_action(self.state(obs)), obs)


def train(agent, episodes, alpha, gamma, eps_start, eps_end, eps_fraction, seed):
    env = agent.env
    rng = np.random.default_rng(seed)
    rewards = []
    decay = int(episodes * eps_fraction)
    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * max(0.0, 1 - ep / decay)
        obs, _ = env.reset(seed = seed + ep if ep == 0 else None)
        s = agent.state(obs)
        total, done = 0.0, False
        while not done:
            a = int(rng.integers(agent.N_ACTIONS)) if rng.random() < eps else int(np.argmax(agent.q[s]))
            obs, r, done, _, _ = env.step(agent.to_env_action(a, obs))
            s2 = agent.state(obs)
            target = r if done else r + gamma * agent.q[s2].max()
            agent.q[s, a] += alpha * (target - agent.q[s, a])
            agent.visits[s] += 1
            s, total = s2, total + r
        rewards.append(total)
    return np.array(rewards)


def plot_curve(rewards, baselines_train, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    smooth = pd.Series(rewards).rolling(200, min_periods = 50).mean()
    fig, ax = plt.subplots(figsize = (7, 4))
    ax.plot(np.arange(len(rewards)) * BikeRebalanceEnv.STEPS_PER_DAY, smooth, color = "#7a4fc9", linewidth = 2, label = "Q-learning")
    for (name, value), color in zip(baselines_train.items(), ["#2a78d6", "#eb6834"]):
        ax.axhline(value, color = color, linestyle = "--", linewidth = 2, label = name)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode reward (200-episode mean)")
    ax.set_title("Tabular Q-learning learning curve on training days", loc = "left")
    ax.legend(frameon = False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi = 200)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type = int, default = 100_000)
    p.add_argument("--alpha", type = float, default = 0.1)
    p.add_argument("--gamma", type = float, default = 0.9)
    p.add_argument("--eps-start", type = float, default = 1.0)
    p.add_argument("--eps-end", type = float, default = 0.05)
    p.add_argument("--eps-fraction", type = float, default = 0.7)
    p.add_argument("--seed", type = int, default = 0)
    p.add_argument("--n-zones", type = int, default = N_ZONES)
    p.add_argument("--split", default = "test", help = "days to evaluate on: test, or train for tuning")
    p.add_argument("--plot", action = "store_true")
    args = p.parse_args()

    env = BikeRebalanceEnv(split = "train")
    zones = make_zones(np.load(DEMAND)["station_ids"], args.n_zones)
    agent = ZoneQAgent(env, zones)
    rewards = train(agent, args.episodes, args.alpha, args.gamma,
                    args.eps_start, args.eps_end, args.eps_fraction, args.seed)

    RESULTS.mkdir(exist_ok = True)
    name = f"qlearning_seed{args.seed}"
    np.savez_compressed(RESULTS / f"{name}.npz", q = agent.q, visits = agent.visits, zones = zones, rewards = rewards)
    with open(RESULTS / f"{name}_rewards.csv", "w", newline = "") as f:
        csv.writer(f).writerows([["episode", "reward"], *enumerate(rewards.round(2))])

    print(f"zones: {np.bincount(zones).tolist()} stations, "
          f"{(agent.visits > 0).sum():,} of {agent.n_states:,} states visited")
    print(f"last 500 training episodes: mean reward {rewards[-500:].mean():.1f}")
    print(summarize(f"{name:>12}", evaluate(agent.policy, split = args.split)))

    if args.plot:
        FIGURES.mkdir(exist_ok = True)
        base = {n: -(r[:, 1] + env.distance_penalty * r[:, 3]).mean()
                for n, r in [("No rebalancing", evaluate(no_rebalance, split = "train")),
                             ("Greedy", evaluate(greedy, split = "train"))]}
        plot_curve(rewards, base, FIGURES / "qlearning_learning_curve.png")


if __name__ == "__main__":
    main()
