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
python data/build_demand.py          # per-station departures/arrivals per 30-min step -> data/processed/demand_top20.npz
```

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

## Tabular Q-learning

```
python train_qlearning.py --seed 0 --plot   # 100k training days (~3 min), prints test-day results, saves figures/qlearning_learning_curve.png
```

Stations are grouped into 3 zones by location (k-means). The state is, per zone, whether a station is near empty (<=15%) or near full (>=85%), plus the van's zone, its load (empty / partial / full), a time-of-day bucket and the weekday flag (6,912 states). Actions are wait, or go to a zone and pick up at its fullest station or drop off at its emptiest (7 actions). Hyperparameters (3 zones, gamma 0.9, 100k episodes) were chosen by results on the training days.
