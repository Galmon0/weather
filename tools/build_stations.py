# -*- coding: utf-8 -*-
"""Генератор каталога станций weather/stations.csv для всех адм. центров субъектов РФ.

Источники:
  * код станции meteo9 — резолвер resolve_city (автодополнение langid=2);
  * координаты/регион/таймзона — geocoding-API Open-Meteo (бесплатный, без ключа).

Запуск (с хоста, не из контейнера):  py tools/build_stations.py
Перезаписывает weather/stations.csv. Города, которые не удалось сопоставить,
выводятся отдельным списком — их добавляем/правим вручную.
"""
import csv
import io
import sys
import time
from pathlib import Path

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from weather.source_meteo9 import resolve_city  # noqa: E402  (после настройки sys.path)

GEO = "https://geocoding-api.open-meteo.com/v1/search"

# IANA-зона -> сдвиг от UTC (часы). Покрывает все российские зоны.
TZ = {
    "Europe/Kaliningrad": 2, "Europe/Moscow": 3, "Europe/Simferopol": 3, "Europe/Volgograd": 3,
    "Europe/Kirov": 3, "Europe/Astrakhan": 4, "Europe/Saratov": 4, "Europe/Samara": 4,
    "Europe/Ulyanovsk": 4, "Asia/Yekaterinburg": 5, "Asia/Omsk": 6, "Asia/Novosibirsk": 7,
    "Asia/Barnaul": 7, "Asia/Tomsk": 7, "Asia/Novokuznetsk": 7, "Asia/Krasnoyarsk": 7,
    "Asia/Irkutsk": 8, "Asia/Chita": 9, "Asia/Yakutsk": 9, "Asia/Khandyga": 9,
    "Asia/Vladivostok": 10, "Asia/Ust-Nera": 10, "Asia/Magadan": 11, "Asia/Sakhalin": 11,
    "Asia/Srednekolymsk": 11, "Asia/Kamchatka": 12, "Asia/Anadyr": 12,
}

# 82 административных центра субъектов РФ
CITIES = [
    "Москва", "Санкт-Петербург",
    "Белгород", "Брянск", "Владимир", "Воронеж", "Иваново", "Калуга", "Кострома", "Курск",
    "Липецк", "Орёл", "Рязань", "Смоленск", "Тамбов", "Тверь", "Тула", "Ярославль",
    "Архангельск", "Великий Новгород", "Вологда", "Калининград", "Мурманск", "Петрозаводск",
    "Псков", "Сыктывкар", "Нарьян-Мар",
    "Астрахань", "Волгоград", "Краснодар", "Майкоп", "Элиста", "Ростов-на-Дону", "Симферополь",
    "Ставрополь", "Махачкала", "Нальчик", "Владикавказ", "Грозный", "Магас", "Черкесск",
    "Казань", "Нижний Новгород", "Самара", "Уфа", "Пермь", "Саратов", "Ульяновск", "Пенза",
    "Оренбург", "Киров", "Чебоксары", "Йошкар-Ола", "Саранск", "Ижевск",
    "Екатеринбург", "Челябинск", "Тюмень", "Курган", "Ханты-Мансийск", "Салехард",
    "Новосибирск", "Омск", "Томск", "Красноярск", "Барнаул", "Кемерово", "Иркутск", "Чита",
    "Улан-Удэ", "Абакан", "Кызыл", "Горно-Алтайск",
    "Владивосток", "Хабаровск", "Благовещенск", "Якутск", "Южно-Сахалинск", "Магадан",
    "Петропавловск-Камчатский", "Анадырь", "Биробиджан",
]


def geocode(name):
    r = requests.get(GEO, params={"name": name, "count": 10, "language": "ru"}, timeout=20)
    res = r.json().get("results") or []
    ru = [x for x in res if x.get("country_code") == "RU"] or res
    ru.sort(key=lambda x: x.get("population") or 0, reverse=True)
    return ru[0] if ru else None


def station_code(name):
    cands = resolve_city(name)
    if not cands:
        return None, None, []
    top = cands[0]
    slug = top["namealt"].split("/")[0]
    return top["id"], slug, cands[:3]


rows, failures = [], []
for i, name in enumerate(CITIES, 1):
    code, slug, cands = station_code(name)
    g = geocode(name)
    if not code or not g:
        failures.append((name, "нет кода meteo9" if not code else "нет geocode"))
        print(f"{i:2d}. {name:24s} ПРОПУСК ({'нет кода' if not code else 'нет geocode'})")
        time.sleep(0.3)
        continue
    tz = TZ.get(g.get("timezone"), round(g["longitude"] / 15))
    rows.append({
        "code": code, "city": name, "region": g.get("admin1", ""), "slug": slug,
        "lat": round(g["latitude"], 4), "lon": round(g["longitude"], 4), "tz_offset": tz,
    })
    alt = "" if len(cands) == 1 else f"  (ещё: {[c['namealt'] for c in cands[1:]]})"
    print(f"{i:2d}. {name:24s} {code:8s} {g.get('timezone',''):20s} "
          f"{round(g['latitude'],2)},{round(g['longitude'],2)}{alt}")
    time.sleep(0.3)

out = PROJ / "weather" / "stations.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["code", "city", "region", "slug", "lat", "lon", "tz_offset"])
    w.writeheader()
    w.writerows(rows)

print(f"\n=== записано {len(rows)} городов в {out} ===")
if failures:
    print(f"=== не удалось ({len(failures)}): ===")
    for name, why in failures:
        print(f"   {name}: {why}")
