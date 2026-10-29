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
python train_dqn.py --seed 0                               # DQN, 1M steps, saves model and test results to results/
python train_dqn.py --seed 0 --double                      # Double DQN
python train_dqn.py --seed 0 --balance-weekends            # sample weekend days as often as weekdays in training
python plot_results.py                                     # needs all 4 variants for seeds 0-1-2; writes figures/
```

All 12 runs in parallel:

```
for seed in 0 1 2; do for flags in "" "--double" "--balance-weekends" "--double --balance-weekends"; do
  OMP_NUM_THREADS=1 python train_dqn.py --seed $seed $flags & done; done; wait
```

## Tabular Q-learning

```
python train_qlearning.py --seed 0 --plot   # 100k training days (~3 min), prints test-day results, saves figures/qlearning_learning_curve.png
```

Stations are grouped into 3 zones by location (k-means). The state is, per zone, whether a station is near empty (<=15%) or near full (>=85%), plus the van's zone, its load (empty / partial / full), a time-of-day bucket and the weekday flag (6,912 states). Actions are wait, or go to a zone and pick up at its fullest station or drop off at its emptiest (7 actions). Hyperparameters (3 zones, gamma 0.9, 100k episodes) were chosen by results on the training days.
