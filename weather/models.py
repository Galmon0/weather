# -*- coding: utf-8 -*-
"""Реестр ML-моделей + бэктест для «конструктора» в приложении.

На лету: выбираем модель, цель (tavg/tmin/tmax) и набор признаков, обучаем на
истории города, сравниваем с персистенсом и предсказываем на завтра.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .forecast import _features

# все доступные признаки конструктора (совпадают с инженерией признаков прогноза)
ALL_FEATURES = ["tavg", "tmin", "tmax", "pressure", "humidity",
                "tavg_lag1", "tavg_lag2", "dpressure", "doy_sin", "doy_cos"]
TARGETS = ["tavg", "tmin", "tmax"]

# линейные модели заворачиваем в StandardScaler (чувствительны к масштабу)
MODELS = {
    "GradientBoosting": lambda: HistGradientBoostingRegressor(
        max_iter=300, max_depth=3, learning_rate=0.05, random_state=0),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=150, max_depth=10, random_state=0, n_jobs=-1),
    "DecisionTree": lambda: DecisionTreeRegressor(max_depth=6, random_state=0),
    "LinearRegression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
}


def make_model(name: str):
    return MODELS[name]()


def backtest(daily: pd.DataFrame, model_name: str, features: list[str],
             target: str = "tavg", eval_days: int = 60) -> dict | None:
    """Обучить выбранную модель на истории до тестового окна, сравнить с персистенсом.

    Возвращает dict с рядами (факт/модель/персистенс), MAE и прогнозом на завтра,
    либо None, если данных мало.
    """
    features = list(features)
    if not features:
        return None
    d = _features(daily)
    d["__target"] = d[target].shift(-1)
    work = d.dropna(subset=features + ["__target"])
    if len(work) < 50:
        return None
    test_n = min(eval_days, len(work) - 40)
    if test_n < 5:
        return None

    tr, te = work.iloc[:-test_n], work.iloc[-test_n:]
    model = make_model(model_name)
    model.fit(tr[features], tr["__target"])
    pred = model.predict(te[features])
    persist = te[target].to_numpy()
    actual = te["__target"].to_numpy()

    # прогноз на завтра: дообучаем на всей доступной истории
    model.fit(work[features], work["__target"])
    last = d.dropna(subset=features).iloc[[-1]]
    tomorrow = float(model.predict(last[features])[0])
    tdate = (last.index[-1] + pd.Timedelta(days=1)).date()

    return {
        "dates": te.index, "pred": pred, "persist": persist, "actual": actual,
        "mae_model": float(mean_absolute_error(actual, pred)),
        "mae_persist": float(mean_absolute_error(actual, persist)),
        "tomorrow": tomorrow, "tomorrow_date": tdate, "n_test": len(te),
    }
