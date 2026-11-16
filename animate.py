import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from stable_baselines3 import DQN, PPO

import train_dqn
import train_ppo
import train_qlearning
from baselines import evaluate, greedy, no_rebalance
from bike_rebalance_env import DEMAND, BikeRebalanceEnv
from plot_results import FIGURES, GRID, INK, MUTED, SURFACE, style

FILL = LinearSegmentedColormap.from_list("fill", ["#eb6834", "#d9d8d3", "#2a78d6"])
LOST = "#e34948"
COLORS = {"No rebalancing": "#2a78d6", "Greedy": "#eb6834", "Q-learning": "#00a3b8", "PPO": "#7a5af5", "DQN": "#e87ba4"}


def run_names(seed):
    return {
        "Q-learning": train_qlearning.run_name(True, True, False, seed),
        "PPO": train_ppo.run_name(False, 0.01, seed),
        "DQN": train_dqn.run_name(False, True, seed),
    }


def typical_weekday(env, seed):
    lost = {"No rebalancing": evaluate(no_rebalance)[:, 1], "Greedy": evaluate(greedy)[:, 1]}
    lost.update({n: np.load(train_dqn.RESULTS / f"{r}_test.npz")["rows"][:, 1] for n, r in run_names(seed).items()})
    weekdays = np.flatnonzero(env.weekdays)
    gaps = sum(np.abs(l[weekdays] - l[weekdays].mean()) / l[weekdays].mean() for l in lost.values())
    return int(weekdays[np.argmin(gaps)])


def policies(env, seed):
    q_name = run_names(seed)["Q-learning"]
    saved = np.load(train_qlearning.RESULTS / f"{q_name}.npz")
    agent = train_qlearning.ZoneQAgent(env, saved["zones"], train_qlearning.FINE_TIME_EDGES)
    agent.q, agent.visits = saved["q"], saved["visits"]
    ppo = PPO.load(train_ppo.RESULTS / f"{train_ppo.run_name(False, 0.01, seed)}.zip")
    dqn = DQN.load(train_dqn.RESULTS / f"{train_dqn.run_name(False, True, seed)}.zip")
    return {
        "No rebalancing": no_rebalance,
        "Greedy": greedy,
        "Q-learning": agent.policy,
        "PPO": train_ppo.ppo_policy(ppo),
        "DQN": train_dqn.dqn_policy(dqn),
    }


def replay(policy, env, day):
    obs, _ = env.reset(options = {"day": day})
    frames = [{"bikes": obs["bikes_per_station"], "van": obs["van_location"], "lost": np.zeros(env.N_STATIONS),
               "total_lost": 0, "km": 0.0}]
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy(obs, env))
        frames.append({"bikes": obs["bikes_per_station"], "van": obs["van_location"], "lost": info["lost_by_station"],
                       "total_lost": frames[-1]["total_lost"] + info["lost"], "km": frames[-1]["km"] + info["km"]})
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type = int, default = 0)
    p.add_argument("--day", type = int, help = "test-day index, default is the weekday where every policy is closest to its weekday average")
    args = p.parse_args()

    env = BikeRebalanceEnv(split = "test")
    day = args.day if args.day is not None else typical_weekday(env, args.seed)
    stations = pd.read_csv(DEMAND.parent / "stations.csv").set_index("station_id").loc[env.station_ids]
    x, y = stations.lon.to_numpy(), stations.lat.to_numpy()

    runs = {name: replay(policy, env, day) for name, policy in policies(env, args.seed).items()}
    steps = len(runs["DQN"])

    fig, axes = plt.subplots(2, 3, figsize = (13, 8.2), facecolor = SURFACE)
    fig.subplots_adjust(left = 0.02, right = 0.98, top = 0.9, bottom = 0.15, wspace = 0.08, hspace = 0.18)
    maps = []
    for ax, name in zip(axes.flat, runs):
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1 / np.cos(np.radians(y.mean())))
        for side in ax.spines.values():
            side.set_visible(False)
        dots = ax.scatter(x, y, s = 170, c = np.zeros(len(x)), cmap = FILL, vmin = 0, vmax = 1,
                          edgecolors = SURFACE, linewidths = 2, zorder = 2)
        path, = ax.plot([], [], color = INK, linewidth = 1, alpha = 0.4, zorder = 1)
        van, = ax.plot([], [], marker = "s", markersize = 9, color = INK, linestyle = "none", zorder = 3)
        ax.plot(0.0, 1.06, marker = "s", markersize = 8, color = COLORS[name], transform = ax.transAxes, clip_on = False)
        title = ax.text(0.04, 1.06, "", transform = ax.transAxes, color = INK, fontsize = 10, va = "center")
        maps.append((name, dots, path, van, title))

    race = axes.flat[5]
    style(race)
    hours = np.arange(steps) / 2
    lines = {n: race.plot([], [], color = COLORS[n], linewidth = 2, label = n)[0] for n in runs}
    race.legend(frameon = False, labelcolor = INK, fontsize = 8, loc = "upper left")
    race.set_xlim(0, 24)
    race.set_ylim(0, max(f[-1]["total_lost"] for f in runs.values()) * 1.08)
    race.set_xticks(range(0, 25, 6), [f"{h:02d}:00" for h in range(0, 25, 6)])
    race.set_ylabel("Lost trips so far", color = MUTED, fontsize = 9)
    race.tick_params(labelsize = 8)
    race.set_title("Lost trips through the day", color = INK, loc = "left", fontsize = 10)
    cursor = race.axvline(0, color = GRID, linewidth = 1)

    bar = fig.colorbar(maps[0][1], cax = fig.add_axes([0.66, 0.04, 0.3, 0.018]), orientation = "horizontal")
    bar.set_label("Station fill: empty → full", color = MUTED, fontsize = 9)
    bar.ax.xaxis.set_label_position("top")
    bar.ax.tick_params(labelsize = 8)
    bar.outline.set_visible(False)
    fig.legend(handles = [
        plt.Line2D([], [], marker = "s", color = INK, linestyle = "none", label = "Van (line: last 4 moves)"),
        plt.Line2D([], [], marker = "o", markerfacecolor = "none", markeredgecolor = LOST, markeredgewidth = 2,
                   linestyle = "none", markersize = 9, label = "Trips lost at this station in the last 30 min"),
    ], loc = "lower left", bbox_to_anchor = (0.01, 0.01), frameon = False, labelcolor = INK, fontsize = 9)
    clock = fig.suptitle("", color = INK, x = 0.01, ha = "left", fontsize = 12)

    def draw(t):
        clock.set_text(f"One test weekday ({env.dates[day]}) replayed under each policy: {t // 2:02d}:{30 * (t % 2):02d}")
        for name, dots, path, van, title in maps:
            f = runs[name][t]
            dots.set_array(f["bikes"] / env.capacity)
            dots.set_edgecolors([LOST if lost > 0 else SURFACE for lost in f["lost"]])
            trail = [runs[name][i]["van"] for i in range(max(0, t - 4), t + 1)]
            path.set_data(x[trail], y[trail])
            van.set_data([x[f["van"]]], [y[f["van"]]])
            title.set_text(f"{name}: {f['total_lost']} lost, {f['km']:.0f} van-km")
            total = [runs[name][i]["total_lost"] for i in range(t + 1)]
            lines[name].set_data(hours[:t + 1], total)
        cursor.set_xdata([hours[t]])

    FuncAnimation(fig, draw, frames = steps).save(FIGURES / "day_replay.gif", writer = PillowWriter(fps = 3), dpi = 80,
                                                  savefig_kwargs = {"facecolor": SURFACE})
    draw(16)
    fig.savefig(FIGURES / "day_replay_0800.png", dpi = 150, facecolor = SURFACE)
    draw(steps - 1)
    fig.savefig(FIGURES / "day_replay_end.png", dpi = 150, facecolor = SURFACE)
    print(env.dates[day], {n: (f[-1]["total_lost"], round(f[-1]["km"])) for n, f in runs.items()})


if __name__ == "__main__":
    main()
