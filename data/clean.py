import json
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "processed"
SF_REGION = "3"
MIN_MINUTES = 1
MAX_MINUTES = 180


def read_month(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
        return pd.read_csv(z.open(name), low_memory=False,
                           usecols=["started_at", "ended_at", "start_station_id", "end_station_id", "rideable_type"])


def load_stations():
    stations = pd.DataFrame(json.loads((RAW / "station_information.json").read_text())["data"]["stations"])
    stations = stations[stations.region_id == SF_REGION]
    return stations[["short_name", "name", "lat", "lon", "capacity"]].rename(columns={"short_name": "station_id"})


def main():
    stations = load_stations()
    sf = set(stations.station_id)
    log = {}

    trips = pd.concat([read_month(p) for p in sorted(RAW.glob("*.zip"))], ignore_index=True)
    log["raw"] = len(trips)

    trips = trips.dropna(subset=["start_station_id", "end_station_id"])
    log["docked"] = len(trips)

    trips = trips[trips.start_station_id.isin(sf) & trips.end_station_id.isin(sf)]
    log["sf"] = len(trips)

    trips["started_at"] = pd.to_datetime(trips.started_at, format="mixed")
    trips["ended_at"] = pd.to_datetime(trips.ended_at, format="mixed")
    minutes = (trips.ended_at - trips.started_at).dt.total_seconds() / 60
    trips = trips[minutes.between(MIN_MINUTES, MAX_MINUTES)]
    log["duration_ok"] = len(trips)

    trips = trips.drop_duplicates().sort_values("started_at").reset_index(drop=True)
    log["final"] = len(trips)

    counts = pd.concat([trips.start_station_id, trips.end_station_id]).value_counts()
    stations = stations.assign(trips=stations.station_id.map(counts).fillna(0).astype(int))
    stations = stations.sort_values("trips", ascending=False)

    OUT.mkdir(exist_ok=True)
    trips.to_parquet(OUT / "trips.parquet", index=False)
    stations.to_csv(OUT / "stations.csv", index=False)

    for step, n in log.items():
        print(f"{step:>12}: {n:>9,}")
    print(f"{trips.started_at.min()} -> {trips.started_at.max()}")


if __name__ == "__main__":
    main()
