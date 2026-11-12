import numpy as np
import gymnasium as gym
from gymnasium import spaces

from bike_rebalance_env import BikeRebalanceEnv


class FlatAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_transfer = 2 * env.unwrapped.MAX_TRANSFER + 1
        self.action_space = spaces.Discrete(env.unwrapped.N_STATIONS * self.n_transfer)

    def action(self, action):
        return np.array(divmod(int(action), self.n_transfer))


class FlatObservation(gym.ObservationWrapper):
    # weekend_time --> separate time-step one-hots for weekdays and weekends, so weekend timing doesn't
    # have to be learned as an interaction with the single weekday flag
    def __init__(self, env, weekend_time = False):
        super().__init__(env)
        e = env.unwrapped
        self.weekend_time = weekend_time
        size = 2 * e.N_STATIONS + 1 + e.STEPS_PER_DAY * (2 if weekend_time else 1) + 1
        self.observation_space = spaces.Box(0.0, 1.0, shape = (size,), dtype = np.float32)

    def observation(self, obs):
        e = self.env.unwrapped
        time = np.eye(e.STEPS_PER_DAY)[obs["time_step"]]
        if self.weekend_time:
            time = np.concatenate([time * obs["weekday"], time * (1 - obs["weekday"])])
        return np.concatenate([
            obs["bikes_per_station"] / e.capacity,
            np.eye(e.N_STATIONS)[obs["van_location"]],
            [obs["van_load"] / e.VAN_CAPACITY],
            time,
            [obs["weekday"]],
        ]).astype(np.float32)


def make_dqn_env(**kwargs):
    return FlatObservation(FlatAction(BikeRebalanceEnv(**kwargs)))
