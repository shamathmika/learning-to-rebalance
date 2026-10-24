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
