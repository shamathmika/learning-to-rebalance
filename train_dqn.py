import argparse
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from baselines import evaluate, summarize
from wrappers import make_dqn_env

RESULTS = Path(__file__).parent / "results"


class DoubleDQN(DQN):
    def train(self, gradient_steps, batch_size = 100):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env = self._vec_normalize_env)
            with th.no_grad():
                next_actions = self.q_net(data.next_observations).argmax(dim = 1, keepdim = True)
                next_q = th.gather(self.q_net_target(data.next_observations), 1, next_actions)
                target = data.rewards + (1 - data.dones) * self.gamma * next_q

            current = th.gather(self.q_net(data.observations), 1, data.actions.long())
            loss = F.smooth_l1_loss(current, target)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude = "tensorboard")
        self.logger.record("train/loss", np.mean(losses))


def run_name(double, seed):
    return f"{'double' if double else 'dqn'}_seed{seed}"


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
    p.add_argument("--double", action = "store_true")
    args = p.parse_args()

    RESULTS.mkdir(exist_ok = True)
    name = run_name(args.double, args.seed)
    env = Monitor(make_dqn_env(split = "train"), str(RESULTS / name))

    algo = DoubleDQN if args.double else DQN
    model = algo(
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

    rows, heat = evaluate(dqn_policy(model), return_heat = True)
    np.savez_compressed(RESULTS / f"{name}_test.npz", rows = rows, heat = heat)
    print(summarize(f"{name:>12}", rows))


if __name__ == "__main__":
    main()
