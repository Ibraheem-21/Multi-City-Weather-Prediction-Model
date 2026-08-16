"""Train Ridge models for one or more cities and save artifacts."""

from __future__ import annotations

import argparse
import sys
import time

from src.cities import CITIES, list_cities
from src.data import load_city_weather
from src.model import train_city


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train next-day TMAX models")
    parser.add_argument(
        "--city",
        action="append",
        dest="cities",
        help="City id (repeatable). Default: all cities.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download Open-Meteo data (ignores cache).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available cities and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for city in list_cities():
            if city.local_csv:
                source = city.local_csv
            elif city.noaa_station:
                source = f"NOAA {city.noaa_station}"
            else:
                source = "Open-Meteo"
            print(f"{city.id:16} {city.name:22} ({source})")
        return 0

    city_ids = args.cities or list(CITIES.keys())
    for i, city_id in enumerate(city_ids):
        if i:
            # Pause between cities to respect Open-Meteo free-tier limits.
            time.sleep(20)
        print(f"\n=== {city_id} ===")
        weather = load_city_weather(city_id, refresh=args.refresh)
        print(
            f"rows={len(weather)} "
            f"range={weather.index.min().date()} -> {weather.index.max().date()}"
        )
        _, result, _, _ = train_city(city_id, weather, persist=True)
        print(
            f"train={result.train_rows} test={result.test_rows} "
            f"RMSE={result.rmse:.2f} F MAE={result.mae:.2f} F"
        )
    print("\nDone. Artifacts written to artifacts/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
