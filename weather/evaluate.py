# -*- coding: utf-8 -*-
"""Бэктест точности прогноза tavg: ошибка модели (GBR) vs персистенс по дням.

Для каждого города: обучаем модель на истории ДО тестового окна и предсказываем
tavg на следующий день для каждого дня окна; сравниваем с фактом. Пишем в
forecast_eval абсолютную ошибку по дням (model='gbr' и 'persistence') — это
питает дашборд точности в приложении.

  docker compose run --rm loader python -m weather.evaluate
"""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy.engine import Engine

from . import db
from .forecast import _load_daily, _features, _model, FEATS, MIN_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate")

EVAL_DAYS = 60   # длина тестового окна (дней)


def run(engine: Engine | None = None, eval_days: int = EVAL_DAYS) -> int:
    engine = engine or db.get_engine()
    stations = db.load_stations_csv()
    recs = []
    for s in stations.itertuples(index=False):
        g = _load_daily(engine, s.code)
        if g.empty:
            continue
        d = _features(g)
        d["target"] = d["tavg"].shift(-1)
        work = d.dropna(subset=FEATS + ["target"])
        if len(work) < MIN_DAYS + 10:
            continue
        test_n = min(eval_days, len(work) - MIN_DAYS)
        if test_n < 5:
            continue

        tr, te = work.iloc[:-test_n], work.iloc[-test_n:]
        model = _model()
        model.fit(tr[FEATS], tr["target"])
        pred = model.predict(te[FEATS])

        for i in range(len(te)):
            row = te.iloc[i]
            eval_date = (te.index[i] + pd.Timedelta(days=1)).date()
            actual = float(row["target"])
            pm, pp = float(pred[i]), float(row["tavg"])   # модель и персистенс
            recs.append({"station_code": s.code, "eval_date": eval_date, "model": "gbr",
                         "predicted": pm, "actual": actual, "abs_error": abs(pm - actual)})
            recs.append({"station_code": s.code, "eval_date": eval_date, "model": "persistence",
                         "predicted": pp, "actual": actual, "abs_error": abs(pp - actual)})

    df = pd.DataFrame(recs)
    written = db.upsert_forecast_eval(engine, df)
    if not df.empty:
        agg = df.groupby("model")["abs_error"].mean()
        log.info("бэктест MAE: gbr=%.2f, персистенс=%.2f (окно ~%d дней, городов с оценкой=%d)",
                 agg.get("gbr", float("nan")), agg.get("persistence", float("nan")),
                 eval_days, df["station_code"].nunique())
    log.info("оценок записано: %d", written)
    return written


if __name__ == "__main__":
    run()
