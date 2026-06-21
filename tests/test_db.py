# -*- coding: utf-8 -*-
"""Тесты приведения типов pandas/numpy -> python для драйвера БД."""
import numpy as np
import pandas as pd

import weather.db as db


def test_py_conversions():
    assert db._py(None) is None
    assert db._py(float("nan")) is None
    assert db._py(pd.NA) is None
    assert db._py(np.int64(5)) == 5 and isinstance(db._py(np.int64(5)), int)
    assert db._py(np.float64(2.5)) == 2.5 and isinstance(db._py(np.float64(2.5)), float)
    assert db._py(pd.Timestamp("2026-06-21 12:00")).year == 2026


def test_records_nan_to_none():
    df = pd.DataFrame({"a": [np.int64(1)], "b": [pd.NA]})
    assert db._records(df) == [{"a": 1, "b": None}]
