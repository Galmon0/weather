# -*- coding: utf-8 -*-
"""Тест инженерии признаков прогноза."""
import pandas as pd

import weather.forecast as fc


def test_features_columns_and_lags():
    idx = pd.date_range("2026-01-01", periods=10, freq="D")
    g = pd.DataFrame({"tavg": range(10), "tmin": range(10), "tmax": range(10),
                      "pressure": [745] * 10, "humidity": [60] * 10}, index=idx)
    d = fc._features(g)
    for col in ["doy_sin", "doy_cos", "tavg_lag1", "tavg_lag2", "dpressure"]:
        assert col in d.columns
    assert d["tavg_lag1"].iloc[1] == 0   # вчерашняя tavg = 0
