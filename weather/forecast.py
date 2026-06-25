# -*- coding: utf-8 -*-
"""Прогноз температуры на завтра (tavg/tmin/tmax) + честная оценка против baseline.

Читает наблюдения из БД, агрегирует в суточные ряды, обучает НЕСКОЛЬКО моделей
(реестр weather/models.py) на каждую цель и пишет прогноз каждой модели в таблицу
forecasts. Рядом пишется baseline-персистенс «завтра = сегодня». Это даёт выбор
модели прямо в таблице прогноза приложения.

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
    # давление/влажность у части станций бывают пустыми (особенно в свежих днях);
    # заполняем, иначе dropna по признакам выкидывает все строки и город выпадает
    # из прогноза целиком либо получает прогноз на устаревшую дату.
    for col in ("pressure", "humidity"):
        d[col] = d[col].ffill().bfill()
        if d[col].isna().all():
            d[col] = 0.0
    d["dpressure"] = d["pressure"].diff().fillna(0.0)   # барическая тенденция
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
    from . import models  # ленивый импорт: реестр моделей зависит от _features (избегаем цикла)

    stations = db.load_stations_csv()
    model_names = list(models.MODELS)
    rows = []
    for st in stations.itertuples(index=False):
        g = _load_daily(engine, st.code)
        if g.empty or len(g.dropna(subset=["tavg"])) < MIN_DAYS:
            continue
        d = _features(g)
        feat_rows = d.dropna(subset=FEATS)
        if feat_rows.empty:
            continue
        last = feat_rows.iloc[[-1]]
        target_date = (last.index[-1] + pd.Timedelta(days=1)).date()

        predicted = {name: {} for name in model_names}   # модель -> {цель: значение}
        persist = {}
        for tgt in TARGETS:
            work = d.copy()
            work["target"] = work[tgt].shift(-1)
            train = work.dropna(subset=FEATS + ["target"])
            if len(train) < MIN_DAYS:
                continue
            persist[tgt] = float(last[tgt].iloc[0])      # персистенс = сегодняшнее значение
            for name in model_names:
                m = models.make_model(name)
                m.fit(train[FEATS], train["target"])
                predicted[name][tgt] = float(m.predict(last[FEATS])[0])

        if not persist:
            continue
        for name in model_names:
            v = predicted[name]
            rows.append({"station_code": st.code, "target_date": target_date, "model": name,
                         "predicted_tavg": v.get("tavg"), "predicted_tmin": v.get("tmin"),
                         "predicted_tmax": v.get("tmax")})
        rows.append({"station_code": st.code, "target_date": target_date, "model": "Персистенс",
                     "predicted_tavg": persist.get("tavg"), "predicted_tmin": persist.get("tmin"),
                     "predicted_tmax": persist.get("tmax")})
        log.info("%-18s %-7s прогноз по %d моделям + персистенс", st.city, st.code, len(model_names))

    written = db.upsert_forecasts(engine, pd.DataFrame(rows))
    log.info("прогнозов записано/обновлено: %d", written)
    return written


if __name__ == "__main__":
    run()
