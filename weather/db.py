# -*- coding: utf-8 -*-
"""Доступ к Postgres: движок, заливка каталога и идемпотентный upsert наблюдений."""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from . import config


def get_engine() -> Engine:
    return create_engine(config.DATABASE_URL, pool_pre_ping=True, future=True)


def wait_for_db(retries: int = 30, delay: float = 2.0) -> Engine:
    """Дождаться готовности Postgres — страховка от гонки при рестарте Docker
    (loader может стартовать раньше, чем поднимется контейнер db)."""
    eng = get_engine()
    for _ in range(retries):
        try:
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            return eng
        except Exception:
            time.sleep(delay)
    raise RuntimeError("БД не поднялась за отведённое время")


def load_stations_csv() -> pd.DataFrame:
    return pd.read_csv(config.STATIONS_CSV)


def _py(v):
    """Привести значение pandas/numpy к python-типу, понятному драйверу (или None)."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _py(v) for k, v in row.items()} for row in df.to_dict("records")]


def upsert_stations(engine: Engine, stations: pd.DataFrame) -> int:
    sql = text("""
        INSERT INTO stations (code, city, region, slug, lat, lon, tz_offset)
        VALUES (:code, :city, :region, :slug, :lat, :lon, :tz_offset)
        ON CONFLICT (code) DO UPDATE SET
            city = EXCLUDED.city, region = EXCLUDED.region, slug = EXCLUDED.slug,
            lat = EXCLUDED.lat, lon = EXCLUDED.lon, tz_offset = EXCLUDED.tz_offset
    """)
    rows = _records(stations)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)


def upsert_observations(engine: Engine, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    sql = text("""
        INSERT INTO observations
            (station_code, ts, temp_c, pressure_mm, humidity_pct, cloud_pct,
             wind_dir, wind_dir_code, wind_speed_min, wind_speed_max, precip_mm, weather_code)
        VALUES
            (:station_code, :ts, :temp_c, :pressure_mm, :humidity_pct, :cloud_pct,
             :wind_dir, :wind_dir_code, :wind_speed_min, :wind_speed_max, :precip_mm, :weather_code)
        ON CONFLICT (station_code, ts) DO UPDATE SET
            temp_c = EXCLUDED.temp_c, pressure_mm = EXCLUDED.pressure_mm,
            humidity_pct = EXCLUDED.humidity_pct, cloud_pct = EXCLUDED.cloud_pct,
            wind_dir = EXCLUDED.wind_dir, wind_dir_code = EXCLUDED.wind_dir_code,
            wind_speed_min = EXCLUDED.wind_speed_min, wind_speed_max = EXCLUDED.wind_speed_max,
            precip_mm = EXCLUDED.precip_mm, weather_code = EXCLUDED.weather_code,
            loaded_at = now()
    """)
    rows = _records(df)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)


def upsert_forecasts(engine: Engine, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    sql = text("""
        INSERT INTO forecasts
            (station_code, target_date, predicted_tmin, predicted_tmax, predicted_tavg, model)
        VALUES
            (:station_code, :target_date, :predicted_tmin, :predicted_tmax, :predicted_tavg, :model)
        ON CONFLICT (station_code, target_date, model) DO UPDATE SET
            predicted_tmin = EXCLUDED.predicted_tmin,
            predicted_tmax = EXCLUDED.predicted_tmax,
            predicted_tavg = EXCLUDED.predicted_tavg,
            created_at = now()
    """)
    rows = _records(df)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)


def upsert_forecast_eval(engine: Engine, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    sql = text("""
        INSERT INTO forecast_eval (station_code, eval_date, model, predicted, actual, abs_error)
        VALUES (:station_code, :eval_date, :model, :predicted, :actual, :abs_error)
        ON CONFLICT (station_code, eval_date, model) DO UPDATE SET
            predicted = EXCLUDED.predicted, actual = EXCLUDED.actual, abs_error = EXCLUDED.abs_error
    """)
    rows = _records(df)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)
