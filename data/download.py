import argparse
import re
import urllib.request
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/baywheels-data"
GBFS_STATIONS = "https://gbfs.lyft.com/gbfs/2.3/bay/en/station_information.json"
RAW = Path(__file__).parent / "raw"


def list_months():
    xml = urllib.request.urlopen(BUCKET).read().decode()
    return {m[1]: m[0] for m in re.finditer(r"(\d{6})-[a-z]+-tripdata[.a-z]*\.zip", xml)}


def fetch(url, dest):
    if dest.exists():
        print(f"skip {dest.name}")
        return
    print(f"get  {dest.name}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="202510")
    p.add_argument("--end", default="202609")
    args = p.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    months = list_months()
    for month in sorted(months):
        if args.start <= month <= args.end:
            fetch(f"{BUCKET}/{months[month]}", RAW / f"{month}.zip")
    fetch(GBFS_STATIONS, RAW / "station_information.json")


if __name__ == "__main__":
    main()
