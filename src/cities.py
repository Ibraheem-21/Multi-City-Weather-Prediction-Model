"""City catalog for multi-city weather modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

# NOAA GHCN-Daily station CSVs:
# https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/
NOAA_GHCN_URL = (
    "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{station}.csv"
)


@dataclass(frozen=True)
class City:
    id: str
    name: str
    latitude: float
    longitude: float
    # Optional bundled NOAA-style CSV already in °F / inches.
    local_csv: str | None = None
    # Optional GHCN station id for remote download (tenths °C / tenths mm).
    noaa_station: str | None = None
    timezone: str = "America/Los_Angeles"


CITIES: dict[str, City] = {
    "oakland": City(
        id="oakland",
        name="Oakland, CA",
        latitude=37.7213,
        longitude=-122.2208,
        local_csv="Oakland_Weather.csv",
        noaa_station="USW00023230",
        timezone="America/Los_Angeles",
    ),
    "toronto": City(
        id="toronto",
        name="Toronto, ON",
        latitude=43.6532,
        longitude=-79.3832,
        noaa_station="CA006158355",
        timezone="America/Toronto",
    ),
    "san_francisco": City(
        id="san_francisco",
        name="San Francisco, CA",
        latitude=37.7749,
        longitude=-122.4194,
        noaa_station="USW00023272",
        timezone="America/Los_Angeles",
    ),
    "new_york": City(
        id="new_york",
        name="New York, NY",
        latitude=40.7128,
        longitude=-74.0060,
        noaa_station="USW00094728",
        timezone="America/New_York",
    ),
    "chicago": City(
        id="chicago",
        name="Chicago, IL",
        latitude=41.8781,
        longitude=-87.6298,
        noaa_station="USW00094846",
        timezone="America/Chicago",
    ),
    "vancouver": City(
        id="vancouver",
        name="Vancouver, BC",
        latitude=49.2827,
        longitude=-123.1207,
        noaa_station="CA001108446",
        timezone="America/Vancouver",
    ),
    "seattle": City(
        id="seattle",
        name="Seattle, WA",
        latitude=47.6062,
        longitude=-122.3321,
        noaa_station="USW00024233",
        timezone="America/Los_Angeles",
    ),
}


def get_city(city_id: str) -> City:
    try:
        return CITIES[city_id]
    except KeyError as exc:
        known = ", ".join(sorted(CITIES))
        raise KeyError(f"Unknown city '{city_id}'. Choose from: {known}") from exc


def list_cities() -> list[City]:
    return list(CITIES.values())
