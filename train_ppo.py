import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor

from baselines import evaluate, summarize
from bike_rebalance_env import BikeRebalanceEnv
from wrappers import FlatObservation

RESULTS = Path(__file__).parent / "results"
LEARNING_RATE = 3e-4
CHECKPOINT_EVERY = 1_000_000


def make_ppo_env(raw_obs = False, weekend_time = False, **kwargs):
    # normalized + one-hot observation, keep the MultiDiscrete action
    # PPO handles (station, transfer) as two factored heads --> no flattening needed, but using same encoding as DQN
    # raw_obs --> env's native Dict observation (raw bike counts) for the normalization ablation
    env = BikeRebalanceEnv(**kwargs)
    return env if raw_obs else FlatObservation(env, weekend_time = weekend_time)


def run_name(balance_weekends, ent_coef, seed, raw_obs = False, lr_decay = False, weekend_time = False, timesteps = 1_000_000):
    tags = [tag for tag, on in [("raw", raw_obs), ("wb", balance_weekends), ("lrdecay", lr_decay), ("wtime", weekend_time)] if on]
    steps = "" if timesteps == 1_000_000 else f"_{timesteps / 1e6:g}M"
    return "ppo" + "".join(f"_{t}" for t in tags) + f"_ent{ent_coef:g}{steps}_seed{seed}"


def ppo_policy(model, raw_obs = False, weekend_time = False):
    flat = make_ppo_env(weekend_time = weekend_time, split = "test")

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
    p.add_argument("--raw-obs", action = "store_true", help = "ablation: raw Dict observation + MultiInputPolicy")
    p.add_argument("--lr-decay", action = "store_true", help = "decay the learning rate linearly to 0 over training")
    p.add_argument("--weekend-time", action = "store_true", help = "separate weekday/weekend time-step one-hots")
    args = p.parse_args()
    if args.raw_obs and args.weekend_time:
        p.error("--weekend-time only applies to the normalized observation")

    RESULTS.mkdir(exist_ok = True)
    name = run_name(args.balance_weekends, args.ent_coef, args.seed, args.raw_obs, args.lr_decay, args.weekend_time, args.timesteps)
    env_kwargs = {"split": "train", "balance_weekends": args.balance_weekends, "raw_obs": args.raw_obs, "weekend_time": args.weekend_time}
    env = make_vec_env(make_ppo_env, n_envs = args.n_envs, seed = args.seed, env_kwargs = env_kwargs)
    env = VecMonitor(env, str(RESULTS / name))

    # one rollout = 8 envs x 512 steps = 4096 steps, about 85 full days
    model = PPO(
        "MultiInputPolicy" if args.raw_obs else "MlpPolicy", env,
        learning_rate = (lambda remaining: LEARNING_RATE * remaining) if args.lr_decay else LEARNING_RATE,
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

    # long runs also save a model every 1M steps so compare_ppo.py can evaluate them on the test days
    callback = None
    if args.timesteps > CHECKPOINT_EVERY:
        callback = CheckpointCallback(save_freq = CHECKPOINT_EVERY // args.n_envs, save_path = str(RESULTS / f"{name}_ckpt"), name_prefix = "ppo")

    start = time.time()
    model.learn(total_timesteps = args.timesteps, callback = callback, progress_bar = False)
    print(f"trained {args.timesteps:,} steps in {time.time() - start:.0f}s")
    # explicit .zip: SB3 only appends it when the name has no suffix and "ent0.01" looks like one
    model.save(RESULTS / f"{name}.zip")

    rows, heat = evaluate(ppo_policy(model, args.raw_obs, args.weekend_time), return_heat = True)
    np.savez_compressed(RESULTS / f"{name}_test.npz", rows = rows, heat = heat)
    print(summarize(f"{name:>22}", rows))


if __name__ == "__main__":
    main()
