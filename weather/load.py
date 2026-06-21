# -*- coding: utf-8 -*-
"""ETL: выкачать наблюдения по всем станциям каталога и залить в Postgres.

Идемпотентно: повторный запуск не плодит дубли (UPSERT по station_code+ts),
поэтому можно безопасно гонять хоть каждый день, хоть вручную.

  python -m weather.load                 # свежая загрузка (текущий месяц)
  python -m weather.load --backfill 24   # догрузить историю за 24 месяца назад
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import time

from . import db
from .source_meteo9 import fetch_month_df, fetch_recent_df, REQUEST_DELAY_S

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("load")


def run_once() -> int:
    """Свежая ежедневная загрузка: текущий месяц по всем станциям."""
    engine = db.get_engine()
    stations = db.load_stations_csv()
    log.info("каталог станций: %d", db.upsert_stations(engine, stations))

    total = 0
    for row in stations.itertuples(index=False):
        try:
            written = db.upsert_observations(engine, fetch_recent_df(row.code))
            total += written
            log.info("%-18s %-7s наблюдений: %d", row.city, row.code, written)
        except Exception as e:  # один город не должен ронять всю загрузку
            log.warning("%-18s %-7s ОШИБКА: %r", row.city, row.code, e)
        time.sleep(REQUEST_DELAY_S)

    log.info("итого записано/обновлено наблюдений: %d", total)
    return total


def _month_range(months_back: int) -> list[tuple[int, int]]:
    y, m = dt.date.today().year, dt.date.today().month
    out = []
    for _ in range(months_back + 1):
        out.append((y, m))
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return list(reversed(out))


def backfill(months_back: int = 24) -> int:
    """Догрузить историю за N месяцев назад (нужно ML-прогнозу для обучения)."""
    engine = db.get_engine()
    stations = db.load_stations_csv()
    db.upsert_stations(engine, stations)

    total = 0
    for row in stations.itertuples(index=False):
        for y, m in _month_range(months_back):
            try:
                total += db.upsert_observations(engine, fetch_month_df(row.code, y, m))
            except Exception as e:
                log.warning("%-18s %-7s %d-%02d ОШИБКА: %r", row.city, row.code, y, m, e)
            time.sleep(REQUEST_DELAY_S)
        log.info("бэкафилл готов: %s", row.city)

    log.info("бэкафилл: всего записано/обновлено %d", total)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Загрузка погоды в Postgres")
    ap.add_argument("--backfill", type=int, metavar="MONTHS",
                    help="догрузить историю за N месяцев назад вместо свежей загрузки")
    args = ap.parse_args()
    if args.backfill:
        backfill(args.backfill)
    else:
        run_once()
