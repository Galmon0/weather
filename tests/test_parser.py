# -*- coding: utf-8 -*-
"""Тесты парсера meteo9: чистка пропусков, осадки, ветер, мок сети."""
import json

import pandas as pd

import weather.source_meteo9 as s


def _payload(items):
    return {"aSynop": items}


def test_non_dict_payload_returns_empty():
    # пустой месяц источник иногда отдаёт как false/скаляр
    assert s.synop_json_to_df(False, "X").empty
    assert s.synop_json_to_df({"aSynop": []}, "X").empty


def test_temp_and_wind_dir_parsed():
    items = [{"DTIME": "01/06/2026 12:00", "T": "15.5", "DD": "5", "FF": "3"}]
    df = s.synop_json_to_df(_payload(items), "X")
    assert len(df) == 1
    assert abs(df["temp_c"].iloc[0] - 15.5) < 1e-9
    assert df["wind_dir"].iloc[0] == "восточный"   # DD=5


def test_precip_sentinel_65535_dropped():
    items = [{"DTIME": "01/06/2026 12:00", "T": "10", "RR": "65535"}]
    df = s.synop_json_to_df(_payload(items), "X")
    assert df["precip_mm"].isna().all()


def test_precip_code_990_dropped():
    items = [{"DTIME": "01/06/2026 12:00", "T": "10", "RR": "990"}]
    df = s.synop_json_to_df(_payload(items), "X")
    assert df["precip_mm"].isna().all()


def test_precip_real_value_kept():
    items = [{"DTIME": "01/06/2026 12:00", "T": "10", "RR": "3.5"}]
    df = s.synop_json_to_df(_payload(items), "X")
    assert abs(df["precip_mm"].iloc[0] - 3.5) < 1e-9


def test_to_float_comma_and_missing():
    out = s._to_float(pd.Series(["12,5", "255", "-"]))
    assert abs(out.iloc[0] - 12.5) < 1e-9
    assert pd.isna(out.iloc[1])   # 255 — маркер пропуска
    assert pd.isna(out.iloc[2])


def test_parse_wind_range():
    lo, hi = s._parse_wind_range(pd.Series(["2-5", "0", "4"]))
    assert (lo.iloc[0], hi.iloc[0]) == (2, 5)
    assert (lo.iloc[1], hi.iloc[1]) == (0, 0)
    assert (lo.iloc[2], hi.iloc[2]) == (4, 4)


def test_fetch_month_df_mocked(monkeypatch):
    payload = _payload([{"DTIME": "01/06/2026 12:00", "T": "20"}])

    class FakeResp:
        text = json.dumps(payload)

        def raise_for_status(self):
            pass

    class FakeSession:
        def post(self, *a, **k):
            return FakeResp()

    monkeypatch.setattr(s, "_session", lambda: FakeSession())
    df = s.fetch_month_df("X", 2026, 6)
    assert len(df) == 1
    assert abs(df["temp_c"].iloc[0] - 20) < 1e-9
