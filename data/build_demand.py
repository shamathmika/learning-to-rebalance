import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path(__file__).parent / "processed"
STEPS_PER_DAY = 48
TEST_DAYS_PER_MONTH = 7


def counts(times, stations, ids, days):
    day = times.dt.normalize()
    keep = day.isin(days)
    d = np.searchsorted(days, day[keep])
    step = ((times[keep] - day[keep]).dt.total_seconds() // (86400 / STEPS_PER_DAY)).astype(int)
    st = stations[keep].map({s: i for i, s in enumerate(ids)}).to_numpy()
    out = np.zeros((len(days), STEPS_PER_DAY, len(ids)), dtype=np.int16)
    np.add.at(out, (d, step.to_numpy(), st), 1)
    return out


def haversine_km(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-stations", type=int, default=20)
    p.add_argument("--start", default="2025-10-01")
    p.add_argument("--end", default="2026-09-30")
    args = p.parse_args()

    trips = pd.read_parquet(PROCESSED / "trips.parquet")
    stations = pd.read_csv(PROCESSED / "stations.csv").head(args.n_stations)
    ids = stations.station_id.to_numpy()
    days = pd.date_range(args.start, args.end, freq="D").to_numpy()

    dep = trips[trips.start_station_id.isin(ids)]
    arr = trips[trips.end_station_id.isin(ids)]
    departures = counts(dep.started_at, dep.start_station_id, ids, days)
    arrivals = counts(arr.ended_at, arr.end_station_id, ids, days)

    dates = pd.DatetimeIndex(days)
    month_end = dates + pd.offsets.MonthEnd(0)
    is_test = (month_end - dates).days < TEST_DAYS_PER_MONTH

    out = PROCESSED / f"demand_top{args.n_stations}.npz"
    np.savez_compressed(
        out,
        departures=departures,
        arrivals=arrivals,
        dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
        weekday=dates.day_of_week < 5,
        is_test=is_test,
        station_ids=ids.astype(str),
        capacity=stations.capacity.to_numpy(),
        distance_km=haversine_km(stations.lat.to_numpy(), stations.lon.to_numpy()),
    )
    print(f"{out.name}: {len(days)} days ({(~is_test).sum()} train / {is_test.sum()} test), {len(ids)} stations")
    print(f"departures/day {departures.sum() / len(days):.0f}, arrivals/day {arrivals.sum() / len(days):.0f}")


if __name__ == "__main__":
    main()
