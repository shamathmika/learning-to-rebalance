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
        - random number for now (replace later)
    """

    metadata = {"render_modes": []}

    STEPS_PER_DAY = 48
    VAN_CAPACITY = 20
    MAX_TRANSFER = 10

    def __init__(self, split = "train", demand_path = DEMAND):
        super().__init__()
        data = np.load(demand_path)
        days = data["is_test"] if split == "test" else ~data["is_test"]
        self.departures = data["departures"][days]
        self.arrivals = data["arrivals"][days]
        self.dates = data["dates"][days]
        self.capacity = data["capacity"].astype(np.int32)
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
        next_station, transfer_raw = action
        transfer = int(transfer_raw) - self.MAX_TRANSFER  # undo the offset

        # updates the environment’s state
        # placeholder --> moves the van and applies the transfer, clipped to valid ranges, but doesn't replay real
        # trip demand yet
        # Swap this for real lost-trip counting once the station list and trip data are ready
        self._van_location = int(next_station)

        transfer = int(np.clip(transfer, -self._van_load, self.VAN_CAPACITY - self._van_load))
        self._bikes_per_station[self._van_location] -= transfer
        self._bikes_per_station = np.clip(self._bikes_per_station, 0, self.capacity).astype(np.int32)
        self._van_load += transfer

        self._time_step += 1
        terminated = self._time_step >= self.STEPS_PER_DAY
        truncated = False

        # placeholder again --> random number standing in for the real reward
        # Swap this for "-1 per lost trip minus a small distance penalty" once lost-trip counting exists?
        reward = -float(self.np_random.integers(0, 3))
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return {
            "bikes_per_station": self._bikes_per_station.copy(),
            "van_location": self._van_location,
            "van_load": self._van_load,
            "time_step": min(self._time_step, self.STEPS_PER_DAY - 1),
        }
