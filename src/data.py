"""Load and cache daily weather for configured cities."""

from __future__ import annotations

import io
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from .cities import DATA_DIR, NOAA_GHCN_URL, ROOT, City, get_city

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_START = "1970-01-01"


def _c_to_f(series: pd.Series) -> pd.Series:
    return series * 9.0 / 5.0 + 32.0


def _mm_to_inches(series: pd.Series) -> pd.Series:
    return series / 25.4


def _normalize_core(frame: pd.DataFrame) -> pd.DataFrame:
    """Return precip / temp_max / temp_min with DatetimeIndex (°F, inches)."""
    core = frame[["precip", "temp_max", "temp_min"]].copy()
    core["precip"] = pd.to_numeric(core["precip"], errors="coerce").fillna(0.0)
    core["temp_max"] = pd.to_numeric(core["temp_max"], errors="coerce")
    core["temp_min"] = pd.to_numeric(core["temp_min"], errors="coerce")
    core = core.ffill().bfill()
    core.index = pd.to_datetime(core.index)
    core = core.sort_index()
    core = core[~core.index.duplicated(keep="last")]
    if core[["temp_max", "temp_min"]].isna().any().any():
        raise ValueError("Weather frame still has missing temperatures after fill.")
    return core


def load_local_csv(path: Path) -> pd.DataFrame:
    """Load NOAA-style CSV with DATE, PRCP, TMAX, TMIN already in °F / inches."""
    weather = pd.read_csv(path, index_col="DATE")
    core = weather[["PRCP", "TMAX", "TMIN"]].copy()
    core.columns = ["precip", "temp_max", "temp_min"]
    return _normalize_core(core)


def fetch_noaa_ghcn(city: City) -> pd.DataFrame:
    """
    Download GHCN-Daily station CSV from NCEI.

    Values are stored as tenths of °C and tenths of mm; convert to °F / inches.
    """
    if not city.noaa_station:
        raise ValueError(f"{city.id} has no NOAA station configured")

    url = NOAA_GHCN_URL.format(station=city.noaa_station)
    response = requests.get(url, timeout=180)
    response.raise_for_status()

    raw = pd.read_csv(
        io.BytesIO(response.content),
        usecols=lambda c: c in {"DATE", "PRCP", "TMAX", "TMIN"},
        low_memory=False,
    )
    if not {"DATE", "PRCP", "TMAX", "TMIN"}.issubset(raw.columns):
        raise RuntimeError(f"Unexpected NOAA columns for {city.noaa_station}: {list(raw.columns)}")

    temps_c_max = pd.to_numeric(raw["TMAX"], errors="coerce") / 10.0
    temps_c_min = pd.to_numeric(raw["TMIN"], errors="coerce") / 10.0
    precip_mm = pd.to_numeric(raw["PRCP"], errors="coerce") / 10.0

    frame = pd.DataFrame(
        {
            "precip": _mm_to_inches(precip_mm).to_numpy(),
            "temp_max": _c_to_f(temps_c_max).to_numpy(),
            "temp_min": _c_to_f(temps_c_min).to_numpy(),
        },
        index=pd.to_datetime(raw["DATE"]),
    )
    frame.index.name = "DATE"
    # Keep modern overlapping history for modeling.
    frame = frame.loc[frame.index >= DEFAULT_START]
    if frame["temp_max"].notna().sum() < 365:
        raise RuntimeError(f"Insufficient NOAA temperature data for {city.id}")
    return _normalize_core(frame)


def _request_open_meteo(
    city: City,
    start_date: str,
    end_date: str,
    *,
    retries: int = 5,
) -> pd.DataFrame:
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(OPEN_METEO_URL, params=params, timeout=120)
            if response.status_code == 429:
                last_error = RuntimeError("Open-Meteo rate limit (429)")
                time.sleep(15 + 5 * attempt)
                continue
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(payload.get("reason", "Open-Meteo error"))
            daily = payload["daily"]
            precip = _mm_to_inches(
                pd.to_numeric(daily["precipitation_sum"], errors="coerce")
            )
            temp_max = _c_to_f(
                pd.to_numeric(daily["temperature_2m_max"], errors="coerce")
            )
            temp_min = _c_to_f(
                pd.to_numeric(daily["temperature_2m_min"], errors="coerce")
            )
            frame = pd.DataFrame(
                {
                    "precip": precip.to_numpy(),
                    "temp_max": temp_max.to_numpy(),
                    "temp_min": temp_min.to_numpy(),
                },
                index=pd.to_datetime(daily["time"]),
            )
            frame.index.name = "DATE"
            if frame["temp_max"].notna().sum() == 0:
                raise RuntimeError(
                    f"Open-Meteo returned no temperatures for {city.id} "
                    f"({start_date} -> {end_date})."
                )
            return frame
        except Exception as exc:  # noqa: BLE001 — retry transient API failures
            last_error = exc
            time.sleep(3 + 2 * attempt)

    raise RuntimeError(
        f"Failed to fetch Open-Meteo data for {city.id}: {last_error}"
    )


def fetch_open_meteo(
    city: City,
    start_date: str = DEFAULT_START,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fallback fetch from Open-Meteo archive API (°F, inches)."""
    if end_date is None:
        end_date = (date.today() - timedelta(days=5)).isoformat()

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    frame = _request_open_meteo(city, start.isoformat(), end.isoformat())
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return _normalize_core(frame)


def cache_path(city_id: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{city_id}_daily.csv"


def _cache_is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, index_col="DATE", parse_dates=True)
        if frame.empty or "temp_max" not in frame.columns:
            return False
        return float(frame["temp_max"].notna().mean()) > 0.9
    except Exception:  # noqa: BLE001
        return False


def load_city_weather(
    city_id: str,
    *,
    refresh: bool = False,
    prefer_local: bool = True,
) -> pd.DataFrame:
    """
    Load cleaned daily weather for a city.

    Preference order:
    1. Bundled local CSV (e.g. Oakland) when prefer_local=True
    2. Valid cached download under data/
    3. NOAA GHCN station CSV
    4. Open-Meteo archive fallback
    """
    city = get_city(city_id)
    cached = cache_path(city_id)

    if prefer_local and city.local_csv:
        local = ROOT / city.local_csv
        if local.exists():
            return load_local_csv(local)

    if not refresh and _cache_is_valid(cached):
        frame = pd.read_csv(cached, index_col="DATE", parse_dates=True)
        return _normalize_core(frame)

    errors: list[str] = []
    if city.noaa_station:
        try:
            frame = fetch_noaa_ghcn(city)
            frame.to_csv(cached)
            return frame
        except Exception as exc:  # noqa: BLE001
            errors.append(f"NOAA: {exc}")

    try:
        frame = fetch_open_meteo(city)
        frame.to_csv(cached)
        return frame
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Open-Meteo: {exc}")

    raise RuntimeError(
        f"Could not load weather for {city_id}. " + " | ".join(errors)
    )
