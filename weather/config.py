# -*- coding: utf-8 -*-
"""Конфигурация проекта: читается из переменных окружения (.env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Путь к каталогу станций (город -> код станции + координаты).
STATIONS_CSV = Path(os.getenv("STATIONS_CSV", BASE_DIR / "stations.csv"))

# Подключение к Postgres. По умолчанию — локальный контейнер из docker-compose.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://weather:weather@localhost:5432/weather",
)

# Источник данных. Сейчас 'meteo9'; задел под смену источника (напр. open-meteo).
WEATHER_SOURCE = os.getenv("WEATHER_SOURCE", "meteo9")

# Время ежедневной загрузки (часы/минуты по таймзоне контейнера).
SCHEDULE_HOUR = int(os.getenv("SCHEDULE_HOUR", "5"))
SCHEDULE_MINUTE = int(os.getenv("SCHEDULE_MINUTE", "30"))
