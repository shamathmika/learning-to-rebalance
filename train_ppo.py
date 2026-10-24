import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from baselines import evaluate, greedy, no_rebalance, summarize
from bike_rebalance_env import BikeRebalanceEnv

MODELS = Path(__file__).parent / "models"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type = int, default = 200_000)
    p.add_argument("--n-envs", type = int, default = 4)
    p.add_argument("--seed", type= int, default = 0)
    args = p.parse_args()

    env = make_vec_env(BikeRebalanceEnv, n_envs = args.n_envs, seed = args.seed, env_kwargs = {"split": "train"})
    model = PPO("MultiInputPolicy", env, seed = args.seed, verbose = 1, device = "cpu")

    start = time.time()
    model.learn(total_timesteps = args.timesteps)
    print(f"trained {args.timesteps:,} steps in {time.time() - start:.0f}s")

    MODELS.mkdir(exist_ok = True)
    model.save(MODELS / "ppo_preliminary")

    def ppo(obs, env):
        return model.predict(obs, deterministic = True)[0]

    print(f"\ntest split ({len(BikeRebalanceEnv(split = 'test').dates)} held-out days)")
    for name, policy in [("no_rebalance", no_rebalance), ("greedy", greedy), ("ppo", ppo)]:
        print(summarize(f"{name:>12}", evaluate(policy)))


if __name__ == "__main__":
    main()
