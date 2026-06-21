# -*- coding: utf-8 -*-
"""Прогноз температуры на завтра (tavg/tmin/tmax) + честная оценка против baseline.

Читает наблюдения из БД, агрегирует в суточные ряды, обучает по модели на каждую
цель (GradientBoosting) и пишет прогноз в таблицу forecasts. Рядом всегда пишется
baseline-персистенс «завтра = сегодня» — чтобы по метрикам было честно видно,
обыгрывает ли модель наивный прогноз.

Замечание (показано прототипом на истории 2025–2026): для горизонта в 1 день
персистенс очень силён (MAE ~1.5–2.8 °C). Чтобы стабильно его обыгрывать, нужно
больше истории (бэкафилл за пару лет, см. weather/load.py --backfill) и признаки
тенденции. Поэтому метрики модели и персистенса всегда логируются вместе.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from . import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("forecast")

FEATS = ["tavg", "tmin", "tmax", "pressure", "humidity",
         "tavg_lag1", "tavg_lag2", "dpressure", "doy_sin", "doy_cos"]
TARGETS = ["tavg", "tmin", "tmax"]   # цель = значение этого поля ЗАВТРА
MIN_DAYS = 40                         # минимум дней, чтобы вообще учить модель


def _load_daily(engine: Engine, code: str) -> pd.DataFrame:
    q = text("""
        SELECT ts, temp_c, pressure_mm, humidity_pct
        FROM observations WHERE station_code = :c ORDER BY ts
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"c": code})
    if df.empty:
        return df
    df = df.dropna(subset=["temp_c"])
    df["date"] = pd.to_datetime(df["ts"]).dt.normalize()
    g = df.groupby("date").agg(
        tavg=("temp_c", "mean"), tmin=("temp_c", "min"), tmax=("temp_c", "max"),
        pressure=("pressure_mm", "mean"), humidity=("humidity_pct", "mean"),
    ).asfreq("D")
    return g


def _features(g: pd.DataFrame) -> pd.DataFrame:
    d = g.copy()
    doy = d.index.dayofyear.to_numpy()
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["tavg_lag1"] = d["tavg"].shift(1)
    d["tavg_lag2"] = d["tavg"].shift(2)
    d["dpressure"] = d["pressure"].diff()          # барическая тенденция
    for col in FEATS + TARGETS:
        if col in d:
            d[col] = d[col].astype("float64")
    return d


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, max_depth=3, learning_rate=0.05, random_state=0)


def _eval(d: pd.DataFrame, target_col: str):
    """Вернуть (mae_модели, mae_персистенса, n) на хвостовом тесте по времени."""
    work = d.copy()
    work["target"] = work[target_col].shift(-1)
    work = work.dropna(subset=FEATS + ["target"])
    n = len(work)
    if n < MIN_DAYS:
        return None
    split = int(n * 0.8)
    tr, te = work.iloc[:split], work.iloc[split:]
    m = _model()
    m.fit(tr[FEATS], tr["target"])
    mae_m = mean_absolute_error(te["target"], m.predict(te[FEATS]))
    mae_p = mean_absolute_error(te["target"], te[target_col])  # персистенс
    return mae_m, mae_p, n


def _predict_tomorrow(d: pd.DataFrame, target_col: str):
    """Обучить на всей истории и предсказать значение target_col на завтра."""
    work = d.copy()
    work["target"] = work[target_col].shift(-1)
    train = work.dropna(subset=FEATS + ["target"])
    if len(train) < MIN_DAYS:
        return None
    m = _model()
    m.fit(train[FEATS], train["target"])
    last = d.dropna(subset=FEATS).iloc[[-1]]
    return float(m.predict(last[FEATS])[0]), last.index[-1]


def run(engine: Engine | None = None) -> int:
    engine = engine or db.get_engine()
    stations = db.load_stations_csv()
    rows = []
    for st in stations.itertuples(index=False):
        g = _load_daily(engine, st.code)
        if g.empty or len(g.dropna(subset=["tavg"])) < MIN_DAYS:
            log.info("%-18s %-7s мало данных для прогноза", st.city, st.code)
            continue
        d = _features(g)

        preds, last_day = {}, None
        for tgt in TARGETS:
            ev = _eval(d, tgt)
            pr = _predict_tomorrow(d, tgt)
            if ev and pr:
                mae_m, mae_p, n = ev
                verdict = "лучше" if mae_m < mae_p else "хуже"
                log.info("%-18s %-4s модель MAE=%.2f / персистенс MAE=%.2f (%s, n=%d)",
                         st.city, tgt, mae_m, mae_p, verdict, n)
                preds[tgt], last_day = pr[0], pr[1]

        if not preds:
            continue
        target_date = (last_day + pd.Timedelta(days=1)).date()
        last_obs = d.loc[last_day]
        rows.append({"station_code": st.code, "target_date": target_date, "model": "gbr",
                     "predicted_tavg": preds.get("tavg"), "predicted_tmin": preds.get("tmin"),
                     "predicted_tmax": preds.get("tmax")})
        # baseline-персистенс на ту же дату — для честного сравнения в БД
        rows.append({"station_code": st.code, "target_date": target_date, "model": "persistence",
                     "predicted_tavg": float(last_obs["tavg"]),
                     "predicted_tmin": float(last_obs["tmin"]),
                     "predicted_tmax": float(last_obs["tmax"])})

    written = db.upsert_forecasts(engine, pd.DataFrame(rows))
    log.info("прогнозов записано/обновлено: %d", written)
    return written


if __name__ == "__main__":
    run()
