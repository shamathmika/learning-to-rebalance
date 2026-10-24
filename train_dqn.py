import argparse
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from baselines import evaluate, summarize
from wrappers import make_dqn_env

RESULTS = Path(__file__).parent / "results"


def dqn_policy(model):
    flat = make_dqn_env(split = "test")

    def policy(obs, env):
        action, _ = model.predict(flat.observation(obs), deterministic = True)
        return flat.env.action(action)
    return policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type = int, default = 1_000_000)
    p.add_argument("--seed", type = int, default = 0)
    args = p.parse_args()

    RESULTS.mkdir(exist_ok = True)
    name = f"dqn_seed{args.seed}"
    env = Monitor(make_dqn_env(split = "train"), str(RESULTS / name))

    model = DQN(
        "MlpPolicy", env,
        learning_rate = 1e-4,
        buffer_size = 100_000,
        learning_starts = 5_000,
        batch_size = 64,
        gamma = 0.99,
        train_freq = 4,
        target_update_interval = 1_000,
        exploration_fraction = 0.3,
        exploration_final_eps = 0.05,
        policy_kwargs = {"net_arch": [256, 256]},
        seed = args.seed,
        verbose = 0,
    )
    model.learn(total_timesteps = args.timesteps, progress_bar = False)
    model.save(RESULTS / name)

    print(summarize(f"{name:>12}", evaluate(dqn_policy(model))))


if __name__ == "__main__":
    main()
