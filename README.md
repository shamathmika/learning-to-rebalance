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
python train_dqn.py --seed 0 --dueling                     # Dueling DQN
python train_dqn.py --seed 0 --distance-penalty 0.3        # reward = -lost - 0.3 * km (default 0.1)
python train_dqn.py --seed 0 --stations 50                 # needs data/build_demand.py --n-stations 50
python plot_results.py                                     # needs every run below; writes figures/ and prints tables
```

All runs in parallel (45 runs, about 45 min on 18 cores):

```
python data/build_demand.py --n-stations 50
for seed in 0 1 2 3 4; do
  for flags in "" "--double" "--balance-weekends" "--double --balance-weekends" "--dueling --balance-weekends"; do
    OMP_NUM_THREADS=1 python train_dqn.py --seed $seed $flags & done; done
for seed in 0 1 2; do
  for flags in "--distance-penalty 0" "--distance-penalty 0.3" "--distance-penalty 1" "--stations 50"; do
    OMP_NUM_THREADS=1 python train_dqn.py --seed $seed --balance-weekends $flags & done; done; wait
```

## PPO

```
python train_ppo.py --seed 0                               # PPO, 1M steps, 8 envs, saves model, training log and test results to results/
python train_ppo.py --seed 0 --balance-weekends            # sample weekend days as often as weekdays in training
python train_ppo.py --seed 0 --ent-coef 0                  # ablation: no entropy bonus
python plot_ppo.py                                         # plots each variant with all of seeds 0-1-2 done; writes figures/ppo_*.png
python analyze_ppo.py --run ppo_wb_ent0.01_seed0           # step-level behavior of one run vs. no rebalancing
```

All 9 runs in parallel:

```
for seed in 0 1 2; do for flags in "" "--balance-weekends" "--ent-coef 0"; do
  OMP_NUM_THREADS=1 python train_ppo.py --seed $seed $flags & done; done; wait
```

PPO sees the same normalized, one-hot observation as DQN but keeps the env's `MultiDiscrete([station, transfer])` action, so the policy has two factored heads instead of one 420-way output. Exploration comes from the entropy bonus (`--ent-coef`) rather than epsilon-greedy.

## Tabular Q-learning

```
python train_qlearning.py --seed 0                                       # 100k training days (~3 min), saves Q-table and test results to results/
python train_qlearning.py --seed 0 --balance-weekends                    # sample weekend days as often as weekdays in training
python train_qlearning.py --seed 0 --fine-time                           # split the midday time bucket at 12pm and 3pm (9,216 states)
python train_qlearning.py --seed 0 --alpha-decay                         # decay the learning rate linearly to 0.005
python train_qlearning.py --seed 0 --eval-only --split train             # score a saved run on training days, no retraining
python plot_qlearning.py                                                 # needs all 6 variants for seeds 0-1-2; writes figures/qlearning_*.png
```

The reported variant (`--balance-weekends --fine-time`) had the lowest lost trips on training days. Results vary a lot by seed, so always compare means over seeds.

## All methods

```
python compare_methods.py                                                # needs DQN, PPO and Q-learning test results in results/; writes figures/method_comparison.png
```

## Reproducing the Q-learning and comparison results

`data/raw/`, `data/processed/` and `results/` aren't in git, so a fresh clone starts with the three data commands above. Then, in bash (zsh doesn't split `$flags` into separate arguments):

```
for seed in 0 1 2; do
  for flags in "" "--balance-weekends" "--fine-time" "--alpha-decay" "--balance-weekends --fine-time" "--balance-weekends --alpha-decay"; do
    python train_qlearning.py --seed $seed $flags                        # trains and scores on test days
    python train_qlearning.py --seed $seed $flags --eval-only --split train
  done
  OMP_NUM_THREADS=1 python train_dqn.py --seed $seed --balance-weekends
  OMP_NUM_THREADS=1 python train_ppo.py --seed $seed
done

python plot_qlearning.py                                                 # Q-learning variants table and figures
python analyze_qlearning.py                                              # training-day scores, curve values, time-of-day losses, seed behavior
python compare_methods.py                                                # all-method tables and figure
```

Q-learning gives the same numbers on a rerun. DQN and PPO come out close, but not identical.

Stations are grouped into 3 zones by location (k-means). The state is, per zone, whether a station is near empty (<=15%) or near full (>=85%), plus the van's zone, its load (empty / partial / full), a time-of-day bucket and the weekday flag (6,912 states). Actions are wait, or go to a zone and pick up at its fullest station or drop off at its emptiest (7 actions). Hyperparameters (3 zones, gamma 0.9, 100k episodes) were chosen by results on the training days.
