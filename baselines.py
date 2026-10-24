import numpy as np

from bike_rebalance_env import BikeRebalanceEnv


def no_rebalance(obs, env):
    return np.array([obs["van_location"], env.MAX_TRANSFER])


def greedy(obs, env):
    fill = obs["bikes_per_station"] / env.capacity
    if obs["van_load"] > 0:
        return np.array([np.argmin(fill), 0])
    return np.array([np.argmax(fill), 2 * env.MAX_TRANSFER])


def evaluate(policy, split = "test"):
    env = BikeRebalanceEnv(split = split)
    rows = []
    for day in range(len(env.dates)):
        obs, _ = env.reset(options = {"day": day})
        lost = km = 0
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy(obs, env))
            lost += info["lost"]
            km += info["km"]
        demand = env.departures[day].sum() + env.arrivals[day].sum()
        rows.append((env.weekdays[day], lost, demand, km))
    return np.array(rows, dtype = float)


def summarize(name, rows):
    out = [name]
    for label, mask in [("all", np.ones(len(rows), bool)), ("weekday", rows[:, 0] == 1), ("weekend", rows[:, 0] == 0)]:
        r = rows[mask]
        out.append(f"{label}: {r[:, 1].mean():6.1f} lost/day ({100 * r[:, 1].sum() / r[:, 2].sum():4.2f}%)")
    out.append(f"{rows[:, 3].mean():6.1f} km/day")
    return " | ".join(out)


if __name__ == "__main__":
    for name, policy in [("no_rebalance", no_rebalance), ("greedy", greedy)]:
        print(summarize(f"{name:>12}", evaluate(policy)))
