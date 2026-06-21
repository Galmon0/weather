# -*- coding: utf-8 -*-
"""FastAPI: JSON-API над данными проекта (та же БД, что и Streamlit).

Запуск (сервис `api` в docker-compose):
    uvicorn weather.api:app --host 0.0.0.0 --port 8000
Документация автоматически на /docs.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query
from sqlalchemy import text

from . import db

app = FastAPI(title="Погода России API", version="1.0")

_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = db.get_engine()
    return _engine


def _q(sql: str, **params) -> list[dict]:
    with engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params).to_dict("records")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stations")
def stations():
    """Каталог городов с координатами."""
    return _q("SELECT code, city, region, lat, lon, tz_offset FROM stations ORDER BY city")


@app.get("/latest")
def latest():
    """Последнее наблюдение по каждому городу (для карты)."""
    return _q("""
        SELECT DISTINCT ON (o.station_code) s.city, o.station_code, o.ts,
               o.temp_c, o.pressure_mm, o.humidity_pct, o.wind_speed_max
        FROM observations o JOIN stations s ON s.code = o.station_code
        ORDER BY o.station_code, o.ts DESC
    """)


@app.get("/forecast")
def forecast(city: Optional[str] = Query(None), model: str = "gbr"):
    """Прогноз на завтра (по городу или все). model: gbr | persistence."""
    sql = """
        SELECT s.city, f.target_date, f.predicted_tmin, f.predicted_tavg, f.predicted_tmax, f.model
        FROM forecasts f JOIN stations s ON s.code = f.station_code
        WHERE f.model = :model AND f.target_date = (SELECT max(target_date) FROM forecasts)
    """
    if city:
        sql += " AND s.city = :city"
    return _q(sql, model=model, city=city)


@app.get("/observations/{code}")
def observations(code: str, days: int = Query(14, ge=1, le=365)):
    """Временной ряд наблюдений по коду станции за N дней."""
    return _q("""
        SELECT ts, temp_c, pressure_mm, humidity_pct, wind_speed_max
        FROM observations
        WHERE station_code = :code AND ts >= now() - make_interval(days => :days)
        ORDER BY ts
    """, code=code, days=days)


@app.get("/accuracy")
def accuracy():
    """Сводка точности из бэктеста: MAE по моделям."""
    return _q("""
        SELECT model, round(avg(abs_error)::numeric, 3) AS mae, count(*) AS n
        FROM forecast_eval GROUP BY model ORDER BY model
    """)
