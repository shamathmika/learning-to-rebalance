from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

DEMAND = Path(__file__).parent / "data" / "processed" / "demand_top20.npz"


class BikeRebalanceEnv(gym.Env):
    """
    Single-van bike-share rebalancing environment.

    State:
        - bikes_per_station: int array, shape --> bikes currently parked at each station
        - van_location: int --> index of the station the van is at
        - van_load: int --> bikes currently sitting in the van
        - time_step: int --> current fixed time step within the simulated day

    Action:
        - next_station: which station to drive to next (can be one of N_STATIONS)
        - transfer: how many bikes to pick up (+) or drop off (-) once there

    Reward:
        - -1 per lost trip minus distance_penalty per km driven
    """

    metadata = {"render_modes": []}

    STEPS_PER_DAY = 48
    VAN_CAPACITY = 20
    MAX_TRANSFER = 10

    def __init__(self, split = "train", demand_path = DEMAND, distance_penalty = 0.1):
        super().__init__()
        data = np.load(demand_path)
        days = data["is_test"] if split == "test" else ~data["is_test"]
        self.departures = data["departures"][days]
        self.arrivals = data["arrivals"][days]
        self.dates = data["dates"][days]
        self.capacity = data["capacity"].astype(np.int32)
        self.distance_km = data["distance_km"]
        self.distance_penalty = distance_penalty
        self.N_STATIONS = len(self.capacity)
        self.MAX_BIKES_PER_STATION = int(self.capacity.max())

        self.observation_space = spaces.Dict({
                "bikes_per_station": spaces.Box(low = 0, high = self.MAX_BIKES_PER_STATION,
                                                shape = (self.N_STATIONS,), dtype = np.int32,),
                "van_location": spaces.Discrete(self.N_STATIONS),
                "van_load": spaces.Discrete(self.VAN_CAPACITY + 1),
                "time_step": spaces.Discrete(self.STEPS_PER_DAY),
            })

        # MultiDiscrete([N_STATIONS, 2*MAX_TRANSFER+1])
        # valid action for the agent --> which station to drive to next, how many bikes to transfer there
        self.action_space = spaces.MultiDiscrete([self.N_STATIONS, 2 * self.MAX_TRANSFER + 1])
        self._bikes_per_station = None
        self._van_location = 0
        self._van_load = 0
        self._time_step = 0
        self._day = 0

    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)
        if options and "day" in options:
            self._day = options["day"]
        else:
            self._day = int(self.np_random.integers(len(self.dates)))

        # TODO: real opening counts are unknown, half full is assumed
        self._bikes_per_station = self.capacity // 2
        self._van_location = 0
        self._van_load = 0
        self._time_step = 0
        return self._get_obs(), {"date": self.dates[self._day]}

    def step(self, action):
        next_station, transfer_raw = int(action[0]), int(action[1])
        transfer = transfer_raw - self.MAX_TRANSFER  # undo the offset

        km = self.distance_km[self._van_location, next_station]
        self._van_location = next_station

        bikes = self._bikes_per_station[next_station]
        free_docks = self.capacity[next_station] - bikes
        transfer = int(np.clip(transfer, -min(self._van_load, free_docks),
                               min(self.VAN_CAPACITY - self._van_load, bikes)))
        self._bikes_per_station[next_station] -= transfer
        self._van_load += transfer

        # TODO: departures and arrivals within a step are netted, which undercounts losses
        t = self._time_step
        level = (self._bikes_per_station + self.arrivals[self._day, t]
                 - self.departures[self._day, t])
        lost_empty = np.maximum(-level, 0)
        lost_full = np.maximum(level - self.capacity, 0)
        self._bikes_per_station = np.clip(level, 0, self.capacity).astype(np.int32)
        lost = int(lost_empty.sum() + lost_full.sum())

        self._time_step += 1
        terminated = self._time_step >= self.STEPS_PER_DAY
        truncated = False

        reward = -lost - self.distance_penalty * km
        info = {"lost": lost, "lost_empty": int(lost_empty.sum()), "lost_full": int(lost_full.sum()),
                "km": km, "transferred": abs(transfer)}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return {
            "bikes_per_station": self._bikes_per_station.copy(),
            "van_location": self._van_location,
            "van_load": self._van_load,
            "time_step": min(self._time_step, self.STEPS_PER_DAY - 1),
        }
