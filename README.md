# learning-to-rebalance
Deep RL for Bike-Share fleet repositioning using real trip data

## Setup

```
pip install -r requirements.txt
```

## Data

```
python data/download.py              # Bay Wheels trips Oct 2025 - Sep 2026 + GBFS stations -> data/raw/
python data/clean.py                 # SF dock-to-dock trips -> data/processed/trips.parquet, stations.csv
python data/build_demand.py          # per-station departures/arrivals per 30-min step + calendar -> data/processed/demand_top20.npz
python data/check_data.py            # sanity checks on stations, trips, holidays, and weather
```

`build_demand.py` also adds per-day `holiday`, `workday` (weekday and not a holiday), `prcp_in`, `tmax_f`, and `rainy`
arrays to the npz, and writes `data/processed/calendar.csv`. `weekday` is unchanged.

- Holidays: US federal holidays from the `holidays` package, plus the day after Thanksgiving, Christmas Eve, and New Year's Eve.
- Weather: NOAA Climate Data Online, Daily Summaries, San Francisco Downtown (USW00023272), Oct 2025 - Oct 2026
  (PRCP, TMAX, TMIN), committed as `data/weather_sf_daily.csv` because it has to be ordered by hand.
  A rainy day has at least 0.1 in of rain.

## Environment and baselines

```
python test_env.py                   # env checks
python baselines.py                  # no-rebalancing and greedy on held-out test days
```

## DQN

```
python train_dqn.py --seed 0         # 1M steps, saves results/dqn_seed0.zip and prints test-day results
python plot_results.py --seed 0      # figures/ learning curve and test lost-trip bars, prints results table
```
