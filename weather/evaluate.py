# -*- coding: utf-8 -*-
"""Бэктест точности прогноза tavg: ошибка КАЖДОЙ модели vs персистенс по дням.

Для каждого города: обучаем каждую модель реестра на истории ДО тестового окна и
предсказываем tavg на следующий день для каждого дня окна; сравниваем с фактом.
Пишем в forecast_eval абсолютную ошибку по дням (по каждой модели + «Персистенс»)
— это питает дашборд точности в приложении (с выбором модели).

  docker compose run --rm loader python -m weather.evaluate
"""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy.engine import Engine

from . import db
from .forecast import _load_daily, _features, FEATS, MIN_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate")

EVAL_DAYS = 60   # длина тестового окна (дней)


def run(engine: Engine | None = None, eval_days: int = EVAL_DAYS) -> int:
    engine = engine or db.get_engine()
    from . import models  # ленивый импорт реестра (избегаем цикла models<->forecast)

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
        actual = te["target"].to_numpy()
        eval_dates = [(te.index[i] + pd.Timedelta(days=1)).date() for i in range(len(te))]

        # персистенс = сегодняшняя tavg
        persist = te["tavg"].to_numpy()
        for i in range(len(te)):
            recs.append({"station_code": s.code, "eval_date": eval_dates[i], "model": "Персистенс",
                         "predicted": float(persist[i]), "actual": float(actual[i]),
                         "abs_error": abs(float(persist[i]) - float(actual[i]))})
        # каждая модель из реестра
        for name in models.MODELS:
            m = models.make_model(name)
            m.fit(tr[FEATS], tr["target"])
            pred = m.predict(te[FEATS])
            for i in range(len(te)):
                recs.append({"station_code": s.code, "eval_date": eval_dates[i], "model": name,
                             "predicted": float(pred[i]), "actual": float(actual[i]),
                             "abs_error": abs(float(pred[i]) - float(actual[i]))})

    df = pd.DataFrame(recs)
    written = db.upsert_forecast_eval(engine, df)
    if not df.empty:
        agg = {k: round(v, 2) for k, v in df.groupby("model")["abs_error"].mean().items()}
        log.info("бэктест MAE по моделям: %s (окно ~%d дн., городов=%d)",
                 agg, eval_days, df["station_code"].nunique())
    log.info("оценок записано: %d", written)
    return written


if __name__ == "__main__":
    run()
