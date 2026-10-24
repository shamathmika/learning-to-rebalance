import numpy as np
from gymnasium.utils.env_checker import check_env

from bike_rebalance_env import BikeRebalanceEnv


def test_check_env():
    check_env(BikeRebalanceEnv(), skip_render_check = True)


def test_bikes_conserved_and_bounded():
    env = BikeRebalanceEnv()
    rng = np.random.default_rng(0)
    for day in range(5):
        obs, _ = env.reset(options = {"day": day})
        total = obs["bikes_per_station"].sum() + obs["van_load"]
        for t in range(env.STEPS_PER_DAY):
            obs, reward, done, _, info = env.step(env.action_space.sample())
            served = env.arrivals[day, t].sum() - info["lost_full"] - (env.departures[day, t].sum() - info["lost_empty"])
            total += served
            assert obs["bikes_per_station"].sum() + obs["van_load"] == total
            assert (obs["bikes_per_station"] >= 0).all() and (obs["bikes_per_station"] <= env.capacity).all()
            assert 0 <= obs["van_load"] <= env.VAN_CAPACITY
            assert reward <= 0
        assert done


def test_pickup_limited_by_station_bikes():
    env = BikeRebalanceEnv()
    env.reset(options = {"day": 0})
    env._bikes_per_station[3] = 2
    before = env._van_load
    _, _, _, _, info = env.step(np.array([3, 2 * env.MAX_TRANSFER]))
    assert env._van_load - before == 2 and info["transferred"] == 2


if __name__ == "__main__":
    test_check_env()
    test_bikes_conserved_and_bounded()
    test_pickup_limited_by_station_bikes()
    print("ok")
