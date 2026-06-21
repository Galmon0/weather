# -*- coding: utf-8 -*-
"""Сопоставление город -> субъект РФ геометрически (point-in-polygon).

Скачивает GeoJSON границ субъектов в app/russia_regions.geojson и строит
app/city_region_map.json: {код станции -> name региона из GeoJSON}. Это нужно
для хороплета (раскраска регионов значением их адм. центра) без матчинга имён.

  py tools/build_region_map.py
"""
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PROJ = Path(__file__).resolve().parent.parent
GEO_URL = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/russia.geojson"
geo_path = PROJ / "app" / "russia_regions.geojson"

raw = urllib.request.urlopen(GEO_URL, timeout=90).read()
geo_path.write_bytes(raw)
gj = json.loads(raw)
print(f"GeoJSON: {len(gj['features'])} регионов -> {geo_path}")


def pip_ring(x, y, ring) -> bool:
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def pip_geom(x, y, geom) -> bool:
    t, c = geom["type"], geom["coordinates"]
    if t == "Polygon":
        return pip_ring(x, y, c[0])
    if t == "MultiPolygon":
        return any(pip_ring(x, y, poly[0]) for poly in c)
    return False


rows = list(csv.DictReader(open(PROJ / "weather" / "stations.csv", encoding="utf-8")))
mapping, misses = {}, []
for r in rows:
    lon, lat = float(r["lon"]), float(r["lat"])
    found = None
    for f in gj["features"]:
        if pip_geom(lon, lat, f["geometry"]):
            found = f["properties"]["name"]
            break
    if found:
        mapping[r["code"]] = found
    else:
        misses.append(r["city"])
    print(f"  {r['city']:26s} -> {found}")

(PROJ / "app" / "city_region_map.json").write_text(
    json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
print(f"\nсопоставлено {len(mapping)}/{len(rows)}; промахи: {misses}")
