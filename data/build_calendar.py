"""Per-day calendar: weekday, US holidays, and San Francisco weather.

Weather comes from NOAA Climate Data Online (Daily Summaries, station USW00023272,
San Francisco Downtown). It has to be ordered by hand, so the CSV is committed at
data/weather_sf_daily.csv instead of being downloaded by download.py.
"""
import argparse
from datetime import timedelta
from pathlib import Path

import holidays
import pandas as pd

DATA = Path(__file__).parent
WEATHER = DATA / "weather_sf_daily.csv"
OUT = DATA / "processed"
RAINY_INCHES = 0.10  # at least ~2.5 mm of rain counts as a rainy day


def holiday_names(dates):
    """Federal US holidays, plus days that behave like holidays for commuting."""
    years = sorted(set(dates.year))
    names = dict(holidays.US(years=years))
    for day, name in list(names.items()):
        if name.startswith("Thanksgiving"):
            names[day + timedelta(days=1)] = "Day after Thanksgiving"
    for year in years:
        names.setdefault(pd.Timestamp(year, 12, 24).date(), "Christmas Eve")
        names.setdefault(pd.Timestamp(year, 12, 31).date(), "New Year's Eve")
    return pd.Series([names.get(d.date(), "") for d in dates], index=dates)


def load_weather(dates):
    w = pd.read_csv(WEATHER, parse_dates=["DATE"]).set_index("DATE").reindex(dates)
    missing = w.TMAX.isna().sum()
    if missing:
        print(f"warning: no weather for {missing} days")
    prcp = w.PRCP
    tmax, tmin = w.TMAX, w.TMIN
    tavg = w.TAVG.fillna((tmax + tmin) / 2)  # TAVG is empty for this station
    return pd.DataFrame({
        "prcp_in": prcp,
        "tmax_f": tmax,
        "tmin_f": tmin,
        "tavg_f": tavg,
        "rainy": prcp >= RAINY_INCHES,
    }, index=dates)


def build_calendar(start, end):
    dates = pd.date_range(start, end, freq="D")
    names = holiday_names(dates)
    cal = pd.DataFrame({
        "weekday": dates.day_of_week < 5,
        "holiday": names != "",
        "holiday_name": names,
    }, index=dates)
    cal["workday"] = cal.weekday & ~cal.holiday
    cal = cal.join(load_weather(dates))
    cal.index.name = "date"
    return cal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-10-01")
    p.add_argument("--end", default="2026-09-30")
    args = p.parse_args()

    cal = build_calendar(args.start, args.end)
    OUT.mkdir(exist_ok=True)
    cal.to_csv(OUT / "calendar.csv")

    print(f"calendar.csv: {len(cal)} days, {cal.workday.sum()} workdays, "
          f"{cal.holiday.sum()} holidays, {cal.rainy.sum()} rainy days")
    for day, row in cal[cal.holiday].iterrows():
        print(f"  {day:%Y-%m-%d %a}  {row.holiday_name}")


if __name__ == "__main__":
    main()
