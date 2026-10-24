"""Sanity checks for the Bay Wheels trips, GBFS stations, and calendar.

Run after download.py and clean.py (build_demand.py is optional):
    python data/check_data.py
Lines starting with FAIL or WARN need a look; everything else is information.
"""
import json
import zipfile
from pathlib import Path

import pandas as pd

from build_calendar import build_calendar

DATA = Path(__file__).parent
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
SF_REGION = "3"
EXPECTED_MONTHS = pd.period_range("2025-10", "2026-09", freq="M").strftime("%Y%m").tolist()
TRIP_COLUMNS = {"ride_id", "rideable_type", "started_at", "ended_at", "start_station_id",
                "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng", "member_casual"}


def report(ok, msg, level="FAIL"):
    print(f"{'ok  ' if ok else level:<5} {msg}")


def check_gbfs():
    print("\n== GBFS station_information ==")
    feed = json.loads((RAW / "station_information.json").read_text())
    st = pd.DataFrame(feed["data"]["stations"])
    print(f"      feed last_updated {pd.to_datetime(feed['last_updated'], unit='s')}, {len(st)} stations")
    st["region_id"] = st.region_id.astype(str)
    print(f"      stations per region_id: {st.region_id.value_counts().to_dict()}")
    sf = st[st.region_id == SF_REGION]
    report(len(sf) > 100, f"{len(sf)} stations in region {SF_REGION} (SF)")
    sf_prefix = sf.short_name.astype(str).str.startswith("SF-").mean()
    report(sf_prefix > 0.9, f"{sf_prefix:.0%} of region {SF_REGION} short_names start with 'SF-'")
    report(sf.capacity.gt(0).all(), f"capacities > 0 (min {sf.capacity.min()}, max {sf.capacity.max()})")
    report(sf.short_name.is_unique, "short_name is unique")
    outside = sf[~(sf.lat.between(37.70, 37.82) & sf.lon.between(-122.52, -122.35))]
    report(outside.empty, f"{len(outside)} region {SF_REGION} stations lie outside San Francisco", "WARN")
    for _, s in outside.iterrows():
        print(f"        {s.short_name:<12} {s['name']} ({s.lat:.4f}, {s.lon:.4f})")
    return set(sf.short_name)


def check_raw_trips(sf_ids):
    print("\n== Raw trip files ==")
    zips = sorted(RAW.glob("*.zip"))
    months = [z.stem for z in zips]
    missing = sorted(set(EXPECTED_MONTHS) - set(months))
    report(not missing, f"{len(zips)} monthly files, missing: {missing or 'none'}")

    frames = []
    for z in zips:
        with zipfile.ZipFile(z) as f:
            name = next(n for n in f.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
            df = pd.read_csv(f.open(name), nrows=0)
            cols = TRIP_COLUMNS - set(df.columns)
            if cols:
                report(False, f"{z.name} is missing columns {sorted(cols)}")
            df = pd.read_csv(f.open(name), engine="pyarrow",
                             usecols=["ride_id", "started_at", "start_station_id", "end_station_id"])
        print(f"      read {z.name}: {len(df):,} trips")
        start = pd.to_datetime(df.started_at, format="mixed")
        in_month = (start.dt.strftime("%Y%m") == z.stem).mean()
        if in_month < 0.95:
            report(False, f"{z.name}: only {in_month:.0%} of trips start in that month")
        frames.append(df[["ride_id", "start_station_id", "end_station_id"]])
    trips = pd.concat(frames, ignore_index=True)
    print(f"      {len(trips):,} raw trips")

    report(trips.ride_id.is_unique, f"ride_id unique ({trips.ride_id.duplicated().sum():,} duplicates)", "WARN")
    docked = trips.dropna(subset=["start_station_id"])
    print(f"      {1 - len(docked) / len(trips):.0%} of trips have no start station (dockless)")

    sf_trips = docked[docked.start_station_id.astype(str).str.startswith("SF-")]
    matched = sf_trips.start_station_id.isin(sf_ids)
    report(matched.mean() > 0.95,
           f"{matched.mean():.1%} of SF trip start IDs match a GBFS short_name", "WARN")
    unmatched = sf_trips[~matched].start_station_id.value_counts().head(10)
    if len(unmatched):
        print("      most common unmatched IDs (trips dropped by clean.py):")
        for sid, n in unmatched.items():
            print(f"        {sid:<12} {n:>7,}")


def check_processed():
    print("\n== Processed trips ==")
    trips = pd.read_parquet(PROCESSED / "trips.parquet")
    stations = pd.read_csv(PROCESSED / "stations.csv")
    cal = build_calendar("2025-10-01", "2026-09-30")

    daily = trips.groupby(trips.started_at.dt.normalize()).size().reindex(cal.index, fill_value=0)
    cal["trips"] = daily

    hours = trips[trips.started_at.dt.day_of_week < 5].started_at.dt.hour.value_counts().reindex(range(24), fill_value=0)
    am, pm = hours.loc[5:11].idxmax(), hours.loc[14:20].idxmax()
    report(am in (7, 8, 9) and pm in (16, 17, 18),
           f"weekday peaks at {am}:00 and {pm}:00 (expect ~8 and ~17; otherwise check time zone)")

    typical = cal[cal.workday].trips.median()
    low = cal[cal.trips < 0.4 * typical]
    report(low.empty, f"{len(low)} days below 40% of a typical workday ({typical:.0f} trips)", "WARN")
    for day, row in low.iterrows():
        why = row.holiday_name or ("rain" if row.rainy else "weekend" if not row.weekday else "?")
        print(f"        {day:%Y-%m-%d %a} {row.trips:>6,} trips  ({why})")

    print("      holidays vs a typical workday:")
    for day, row in cal[cal.holiday].iterrows():
        print(f"        {day:%Y-%m-%d} {row.holiday_name:<38} {row.trips / typical:>5.0%}")

    dry = cal[cal.workday & ~cal.rainy].trips.median()
    wet = cal[cal.workday & cal.rainy].trips.median()
    print(f"      rainy workdays have {wet / dry:.0%} of the trips of dry workdays")

    top = stations.head(20)
    report(top.capacity.gt(0).all(), "top-20 stations all have a capacity")
    print("      top 20 stations:")
    for _, s in top.iterrows():
        print(f"        {s.station_id:<10} cap {s.capacity:>3}  {s.trips:>7,} trips  {s['name']}")


def main():
    sf_ids = check_gbfs()
    check_raw_trips(sf_ids)
    if (PROCESSED / "trips.parquet").exists():
        check_processed()
    else:
        print("\nskip processed checks: run data/clean.py first")


if __name__ == "__main__":
    main()
