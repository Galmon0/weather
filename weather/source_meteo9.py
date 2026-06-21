# -*- coding: utf-8 -*-
"""
Источник данных: meteo9.ru — синоптические наблюдения с шагом 3 часа.

Рефактор из parsing.ipynb. Что изменено по сравнению с ноутбуком:
  * добавлен ОБЯЗАТЕЛЬНЫЙ параметр `lfile` в запрос архива. Без него сервер
    отвечает `Error #SDM004;` (формат запроса подсмотрен в их main.js:
    data="id="+id+"&month="+m+"&year="+y+"&lfile="+lfile);
  * 65535 распознаётся как маркер пропуска (встречается в осадках);
  * добавлены temp_now() — «температура сейчас» и resolve_city() — резолвер
    станций через эндпоинт автодополнения (langid=2 = русский).

ВНИМАНИЕ по этике: robots.txt сайта запрещает /responses/. Использовать
аккуратно: низкая частота (раз в день), кэширование в БД (не перезапрашивать
уже загруженное), честный User-Agent, задержки между запросами.
"""
from __future__ import annotations

import json
import re
import time
import datetime as dt
from typing import Optional

import requests
import pandas as pd

BASE_URL = "https://meteo9.ru"
ARCHIVE_URL = f"{BASE_URL}/responses/reJsonSynopDataMeteo.php"
TEMPNOW_URL = f"{BASE_URL}/responses/reJsonTempNowMeteo.php"
POINT_URL = f"{BASE_URL}/responses/reJsonPointMeteo.php"

USER_AGENT = "Mozilla/5.0 (weather-map educational project)"
REQUEST_DELAY_S = 1.0  # вежливая пауза между запросами к сайту

WIND_DIR_MAP = {
    0: "штиль", 1: "северный", 2: "северо-северо-восточный", 3: "северо-восточный",
    4: "востоко-северо-восточный", 5: "восточный", 6: "востоко-юго-восточный",
    7: "юго-восточный", 8: "юго-юго-восточный", 9: "южный", 10: "юго-юго-западный",
    11: "юго-западный", 12: "западо-юго-западный", 13: "западный",
    14: "западо-северо-западный", 15: "северо-западный", 16: "северо-северо-западный",
    17: "переменный", 99: "переменный",
}

# Маркеры «нет данных». 65535 добавлен относительно ноутбука (осадки).
MISSING_MARKERS = {"255", 255, "32767", 32767, "65535", 65535, "", None}

_SESSION: Optional[requests.Session] = None


def _session() -> requests.Session:
    """Ленивая сессия с прогревом (получаем PHPSESSID c главной страницы)."""
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        s.get(BASE_URL, timeout=30)
        _SESSION = s
    return _SESSION


# --------------------------------------------------------------------------- #
#  Помощники чистки (из ноутбука, без изменений по смыслу)
# --------------------------------------------------------------------------- #
def _clean_scalar(x):
    if pd.isna(x):
        return pd.NA
    if isinstance(x, str):
        x = x.strip()
    if x in MISSING_MARKERS:
        return pd.NA
    return x


def _series_or_na(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].map(_clean_scalar)
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")


def _to_float(s: pd.Series) -> pd.Series:
    s = s.map(_clean_scalar).astype("string")
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^\d\.\-]+", "", regex=True)
    s = s.replace({"": pd.NA, "-": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(_to_float(s), errors="coerce").astype("Int64")


def _find_first_existing_column(df: pd.DataFrame, exact=(), patterns=()):
    cols = [str(c) for c in df.columns]
    for name in exact:
        if name in cols:
            return name
    for pattern in patterns:
        for col in cols:
            if re.fullmatch(pattern, col, flags=re.I):
                return col
    return None


def _parse_wind_range(series: pd.Series):
    s = series.astype("string").str.replace(",", ".", regex=False)
    parts = s.str.extract(
        r"^\s*(?P<min>-?\d+(?:\.\d+)?)\s*(?:-\s*(?P<max>-?\d+(?:\.\d+)?))?\s*$"
    )
    wind_min = pd.to_numeric(parts["min"], errors="coerce")
    wind_max = pd.to_numeric(parts["max"], errors="coerce").fillna(wind_min)
    return wind_min, wind_max


# --------------------------------------------------------------------------- #
#  Сетевые запросы
# --------------------------------------------------------------------------- #
def fetch_archive_json(code: str, year: int, month: int, lfile: int = 0) -> dict:
    """Сырой JSON архива за месяц для станции `code`."""
    s = _session()
    r = s.post(
        ARCHIVE_URL,
        data={"id": code, "month": str(month), "year": str(year), "lfile": str(lfile)},
        headers={"X-Requested-With": "XMLHttpRequest", "Origin": BASE_URL, "Referer": f"{BASE_URL}/"},
        timeout=30,
    )
    r.raise_for_status()
    body = r.text
    if body[:5] == "Error":  # сервер отдаёт строки вида 'Error #SDM004;'
        raise RuntimeError(f"meteo9 archive error для {code} {year}-{month:02d}: {body!r}")
    return json.loads(body)


def synop_json_to_df(payload: dict, code: str) -> pd.DataFrame:
    """Сырой JSON -> аккуратный DataFrame, готовый к заливке в БД."""
    if not isinstance(payload, dict):   # пустой месяц сервер иногда отдаёт как false/JSON-скаляр
        return pd.DataFrame()
    items = payload.get("aSynop", [])
    raw = pd.DataFrame(items)
    if raw.empty:
        return pd.DataFrame()

    ts = pd.to_datetime(
        _series_or_na(raw, "DTIME"), format="%d/%m/%Y %H:%M", errors="coerce", dayfirst=True
    )

    pressure_mm = _to_float(_series_or_na(raw, "P0"))
    if pressure_mm.isna().all() and "4" in raw.columns:
        pressure_mm = _to_float(_series_or_na(raw, "4"))
    pressure_mm = pressure_mm.where(pressure_mm <= 2000, pressure_mm / 10)

    cloud_raw = _to_float(_series_or_na(raw, "N"))
    cloud_pct = cloud_raw.where(cloud_raw > 10, cloud_raw * 10)
    cloud_pct = cloud_pct.where(cloud_pct <= 100)

    wind_dir_code = _to_int(_series_or_na(raw, "DD"))
    wind_dir = wind_dir_code.map(lambda x: WIND_DIR_MAP.get(x, pd.NA) if pd.notna(x) else pd.NA)
    wind_min, wind_max = _parse_wind_range(_series_or_na(raw, "FF").astype("string"))

    precip_col = _find_first_existing_column(
        raw, exact=("RR", "RRR", "R", "precip_mm", "precip"),
        patterns=(r"R{1,3}", r".*precip.*", r".*osad.*"),
    )
    if precip_col:
        precip_mm = _to_float(_series_or_na(raw, precip_col))
        # Осадки на этом источнике «грязные»: 65535 — нет данных (снято выше),
        # а 990-999 — кодовые значения SYNOP (след/малые суммы), не миллиметры.
        # Чтобы не отравлять анализ, всё >= 900 считаем неизвестным.
        # TODO: при необходимости декодировать RRR-группу корректно.
        precip_mm = precip_mm.where(precip_mm < 900)
    else:
        precip_mm = pd.Series([pd.NA] * len(raw))

    df = pd.DataFrame({
        "station_code": code,
        "ts": ts,
        "temp_c": _to_float(_series_or_na(raw, "T")),
        "pressure_mm": pressure_mm,
        "humidity_pct": _to_int(_series_or_na(raw, "U")),
        "cloud_pct": cloud_pct,
        "wind_dir": wind_dir.astype("string"),
        "wind_dir_code": wind_dir_code,
        "wind_speed_min": wind_min,
        "wind_speed_max": wind_max,
        "precip_mm": precip_mm,
        "weather_code": _to_int(_series_or_na(raw, "WW")),
    })
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


def fetch_month_df(code: str, year: int, month: int) -> pd.DataFrame:
    """Аккуратный DataFrame наблюдений за конкретный месяц."""
    return synop_json_to_df(fetch_archive_json(code, year, month), code)


def fetch_recent_df(code: str, today: Optional[dt.date] = None) -> pd.DataFrame:
    """
    Свежие данные для ежедневной загрузки: текущий месяц, а в начале месяца —
    ещё и предыдущий (чтобы не было дыр на стыке). Дубли убираются по ts.
    """
    today = today or dt.date.today()
    months = [(today.year, today.month)]
    if today.day <= 3:
        prev = (today.replace(day=1) - dt.timedelta(days=1))
        months.insert(0, (prev.year, prev.month))

    frames = []
    for y, m in months:
        frames.append(fetch_month_df(code, y, m))
        time.sleep(REQUEST_DELAY_S)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["station_code", "ts"]).reset_index(drop=True)


def temp_now(code: str) -> Optional[float]:
    """Температура «прямо сейчас» по станции (эндпоинт reJsonTempNowMeteo)."""
    s = _session()
    r = s.post(TEMPNOW_URL, data={"id": code},
               headers={"X-Requested-With": "XMLHttpRequest", "Referer": f"{BASE_URL}/"}, timeout=20)
    try:
        val = json.loads(r.text)[0][1][0]["temperature_now"]
        return float(str(val).replace("+", "").strip())
    except Exception:
        return None


def resolve_city(query: str, langid: int = 2) -> list[dict]:
    """
    Поиск станций по названию города (langid=2 — русский).
    Возвращает список словарей {'id', 'name', 'namealt'}.
    Удобно для расширения каталога stations.csv новыми городами.
    """
    s = _session()
    r = s.get(POINT_URL,
              params={"langid": str(langid), "q": query, "limit": "20", "timestamp": int(time.time() * 1000)},
              headers={"X-Requested-With": "XMLHttpRequest", "Referer": f"{BASE_URL}/"}, timeout=20)
    r.encoding = "utf-8"
    if not r.text.strip().startswith("["):
        return []
    return json.loads(r.text)


if __name__ == "__main__":
    # быстрый дымовой тест
    df = fetch_month_df("Se7X", 2026, 6)
    print(df.tail(6).to_string())
    print("temp_now Москва:", temp_now("Se7X"))
