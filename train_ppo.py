import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor

from baselines import evaluate, summarize
from bike_rebalance_env import BikeRebalanceEnv
from wrappers import FlatObservation

RESULTS = Path(__file__).parent / "results"


def make_ppo_env(**kwargs):
    # normalized + one-hot observation, keep the MultiDiscrete action
    # PPO handles (station, transfer) as two factored heads --> no flattening needed, but using same encoding as DQN
    return FlatObservation(BikeRebalanceEnv(**kwargs))


def run_name(balance_weekends, ent_coef, seed):
    return f"ppo{'_wb' if balance_weekends else ''}_ent{ent_coef:g}_seed{seed}"


def ppo_policy(model, raw_obs = False):
    flat = make_ppo_env(split = "test")

    def policy(obs, env):
        action, _ = model.predict(obs if raw_obs else flat.observation(obs), deterministic = True)
        return action
    return policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type = int, default = 1_000_000)
    p.add_argument("--n-envs", type = int, default = 8)
    p.add_argument("--seed", type = int, default = 0)
    p.add_argument("--balance-weekends", action = "store_true")
    p.add_argument("--ent-coef", type = float, default = 0.01)
    args = p.parse_args()

    RESULTS.mkdir(exist_ok = True)
    name = run_name(args.balance_weekends, args.ent_coef, args.seed)
    env = make_vec_env(make_ppo_env, n_envs = args.n_envs, seed = args.seed, env_kwargs = {"split": "train", "balance_weekends": args.balance_weekends})
    env = VecMonitor(env, str(RESULTS / name))

    # one rollout = 8 envs x 512 steps = 4096 steps, about 85 full days
    model = PPO(
        "MlpPolicy", env,
        learning_rate = 3e-4,
        n_steps = 512,
        batch_size = 256,
        n_epochs = 10,
        gamma = 0.99,
        gae_lambda = 0.95,
        clip_range = 0.2,
        ent_coef = args.ent_coef,
        policy_kwargs = {"net_arch": [256, 256]},
        seed = args.seed,
        verbose = 0,
        device = "cpu",
    )

    # entropy, approx_kl, clip_fraction, explained_variance for each update
    model.set_logger(configure(str(RESULTS / f"{name}_log"), ["csv"]))

    start = time.time()
    model.learn(total_timesteps = args.timesteps, progress_bar = False)
    print(f"trained {args.timesteps:,} steps in {time.time() - start:.0f}s")
    model.save(RESULTS / name)

    rows, heat = evaluate(ppo_policy(model), return_heat = True)
    np.savez_compressed(RESULTS / f"{name}_test.npz", rows = rows, heat = heat)
    print(summarize(f"{name:>22}", rows))


if __name__ == "__main__":
    main()
