# -*- coding: utf-8 -*-
"""Ежедневный запуск: загрузка свежих данных + прогноз + оценка точности.

Сервис `loader` в docker-compose. При старте делает один прогон сразу,
дальше — ежедневно по расписанию (SCHEDULE_HOUR/SCHEDULE_MINUTE).
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from . import config
from .load import run_once
from .forecast import run as run_forecast
from .evaluate import run as run_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("scheduler")


def daily_job() -> None:
    run_once()
    try:
        run_forecast()
    except Exception as e:  # прогноз не должен ронять загрузку
        log.warning("прогноз не удался: %r", e)
    try:
        run_evaluate()
    except Exception as e:  # оценка точности не критична
        log.warning("оценка точности не удалась: %r", e)


def main() -> None:
    log.info("стартовый прогон...")
    try:
        daily_job()
    except Exception as e:
        log.warning("стартовый прогон не удался (продолжаю по расписанию): %r", e)

    sched = BlockingScheduler(timezone="Europe/Moscow")
    sched.add_job(
        daily_job, "cron",
        hour=config.SCHEDULE_HOUR, minute=config.SCHEDULE_MINUTE,
        id="daily_job", misfire_grace_time=3600, coalesce=True,
    )
    log.info("планировщик: ежедневно в %02d:%02d (Europe/Moscow)",
             config.SCHEDULE_HOUR, config.SCHEDULE_MINUTE)
    sched.start()


if __name__ == "__main__":
    main()
