# -*- coding: utf-8 -*-
"""Streamlit: карта России (метрики, точки/хороплет, анимация) + прогноз + точность + графики."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import text

from weather import db, forecast, models

st.set_page_config(page_title="Погода России", layout="wide")
APP_DIR = Path(__file__).resolve().parent

# метрика -> (колонка в БД, цветовая шкала)
METRICS = {
    "Температура, °C": ("temp_c", "RdYlBu_r"),
    "Давление, мм рт.ст.": ("pressure_mm", "Viridis"),
    "Влажность, %": ("humidity_pct", "Blues"),
    "Ветер, м/с": ("wind_speed_max", "Greens"),
}
# рамка карты под всю Россию (Калининград 20°E … Чукотка 177°E)
GEO_RU = dict(projection_type="natural earth", lataxis_range=[40, 78], lonaxis_range=[18, 180],
              showcountries=True, showland=True, landcolor="#f5f5f0", showframe=False)


@st.cache_resource
def get_engine():
    return db.get_engine()


@st.cache_data(ttl=300)
def load_stations() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM stations ORDER BY city", get_engine())


@st.cache_data(ttl=300)
def load_latest() -> pd.DataFrame:
    q = """
        SELECT DISTINCT ON (station_code)
               station_code, ts, temp_c, pressure_mm, humidity_pct, wind_speed_max, wind_dir, precip_mm
        FROM observations
        ORDER BY station_code, ts DESC
    """
    return pd.read_sql(q, get_engine())


@st.cache_data(ttl=300)
def load_forecasts() -> pd.DataFrame:
    q = """
        SELECT f.station_code, s.city, f.target_date, f.model,
               f.predicted_tmin, f.predicted_tavg, f.predicted_tmax
        FROM forecasts f
        JOIN stations s ON s.code = f.station_code
        WHERE f.target_date = (SELECT max(target_date) FROM forecasts)
    """
    return pd.read_sql(q, get_engine())


@st.cache_data(ttl=300)
def load_eval() -> pd.DataFrame:
    return pd.read_sql("SELECT station_code, eval_date, model, abs_error FROM forecast_eval",
                       get_engine())


@st.cache_data(ttl=300)
def load_daily_city(code: str) -> pd.DataFrame:
    """Суточный ряд (tavg/tmin/tmax/pressure/humidity) для конструктора моделей."""
    return forecast._load_daily(get_engine(), code)


@st.cache_data(ttl=300)
def load_series(code: str, days: int) -> pd.DataFrame:
    q = text("""
        SELECT ts, temp_c, pressure_mm, humidity_pct
        FROM observations WHERE station_code = :c AND ts >= :since ORDER BY ts
    """)
    since = dt.datetime.now() - dt.timedelta(days=days)
    with get_engine().connect() as conn:
        return pd.read_sql(q, conn, params={"c": code, "since": since})


@st.cache_data(ttl=300)
def load_daily_metric(col: str, days: int) -> pd.DataFrame:
    if col not in {c for c, _ in METRICS.values()}:
        return pd.DataFrame()
    q = text(f"""
        SELECT station_code, ts::date AS d, avg({col}) AS val
        FROM observations WHERE ts >= :since GROUP BY station_code, ts::date ORDER BY d
    """)
    since = dt.datetime.now() - dt.timedelta(days=days)
    with get_engine().connect() as conn:
        return pd.read_sql(q, conn, params={"since": since})


@st.cache_data
def load_region_map() -> dict:
    p = APP_DIR / "city_region_map.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def load_geojson() -> dict | None:
    p = APP_DIR / "russia_regions.geojson"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def selected_codes_from_map(event, df: pd.DataFrame) -> list[str]:
    try:
        sel = event["selection"] if isinstance(event, dict) else getattr(event, "selection", None)
        pts = (sel["points"] if isinstance(sel, dict) else getattr(sel, "points", [])) or []
    except Exception:
        return []
    codes = []
    for p in pts:
        idx = p.get("point_index", p.get("point_number")) if isinstance(p, dict) else None
        if idx is not None and 0 <= idx < len(df):
            codes.append(df.iloc[idx]["code"])
    return list(dict.fromkeys(codes))


st.title("🌡️ Погода России — карта, прогноз, анализ")

try:
    stations = load_stations()
except Exception as e:
    st.error(f"Нет связи с базой данных: {e}")
    st.stop()
if stations.empty:
    st.warning("Каталог пуст. Запустите загрузку: `docker compose run --rm loader python -m weather.load`")
    st.stop()

latest = load_latest()
m = stations.merge(latest, left_on="code", right_on="station_code", how="left")

# --- Карта ---
st.subheader("Карта")
c1, c2 = st.columns([3, 2])
metric_label = c1.radio("Метрика", list(METRICS), index=0, horizontal=True)
view = c2.radio("Вид", ["Точки", "Регионы"], index=0, horizontal=True)
metric_col, cscale = METRICS[metric_label]

mdata = m.dropna(subset=[metric_col]).reset_index(drop=True)
sel_codes: list[str] = []

if mdata.empty:
    st.info("Нет данных по выбранной метрике.")
elif view == "Точки":
    fig = px.scatter_geo(mdata, lat="lat", lon="lon", color=metric_col, text="city",
                         hover_name="city", color_continuous_scale=cscale,
                         labels={metric_col: metric_label})
    fig.update_traces(marker=dict(size=11, line=dict(width=0.7, color="#555")),
                      textposition="top center", textfont=dict(color="#333333", size=9))
    fig.update_geos(**GEO_RU)
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
    event = st.plotly_chart(fig, width="stretch", key="map",
                            on_select="rerun", selection_mode=["points", "box", "lasso"])
    sel_codes = selected_codes_from_map(event, mdata)
    st.caption(f"Выдели города кликом/рамкой/лассо — прогноз и графики отфильтруются. "
               f"Последнее наблюдение: {m['ts'].max()}")
else:
    gj, rmap = load_geojson(), load_region_map()
    if not gj or not rmap:
        st.info("Границы регионов не найдены. Сгенерируйте: `py tools/build_region_map.py`")
    else:
        dfr = mdata.copy()
        dfr["region_geo"] = dfr["code"].map(rmap)
        dfr = dfr.dropna(subset=["region_geo"])
        figc = px.choropleth(dfr, geojson=gj, locations="region_geo",
                             featureidkey="properties.name", color=metric_col,
                             hover_name="city", color_continuous_scale=cscale,
                             labels={metric_col: metric_label})
        figc.update_geos(fitbounds="locations", visible=False)
        figc.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figc, width="stretch")
        st.caption("Заливка субъекта = значение его областного центра. "
                   "(Крым без данных в этом наборе границ.)")

# Анимация по дням
with st.expander("▶ Анимация по дням"):
    adays = st.slider("Период анимации, дней", 5, 30, 14)
    dm = load_daily_metric(metric_col, adays)
    if dm.empty:
        st.info("Нет данных для анимации.")
    else:
        dm = dm.merge(stations[["code", "city", "lat", "lon"]], left_on="station_code", right_on="code")
        dm["d"] = dm["d"].astype(str)
        dm = dm.sort_values("d")
        figa = px.scatter_geo(dm, lat="lat", lon="lon", color="val", hover_name="city",
                              animation_frame="d", color_continuous_scale=cscale,
                              range_color=[dm["val"].min(), dm["val"].max()],
                              labels={"val": metric_label})
        figa.update_traces(marker=dict(size=10, line=dict(width=0.5, color="#555")))
        figa.update_geos(**GEO_RU)
        figa.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figa, width="stretch")

# --- Прогноз на завтра ---
st.subheader("Прогноз на завтра")
extra_cities = st.multiselect("Города (или выдели на карте выше)", stations["city"].tolist(), default=[])
extra_codes = stations.loc[stations["city"].isin(extra_cities), "code"].tolist()
chosen = list(dict.fromkeys(sel_codes + extra_codes))

try:
    fc = load_forecasts()
except Exception:
    fc = pd.DataFrame()
if fc.empty:
    st.info("Прогноз ещё не рассчитан (идёт бэкафилл/первый прогон).")
else:
    gbr = fc[fc["model"] == "gbr"].copy()
    if chosen:
        gbr = gbr[gbr["station_code"].isin(chosen)]
        names = ", ".join(stations.loc[stations["code"].isin(chosen), "city"])
        st.caption(f"Выбрано: **{len(chosen)}** ({names}). Модель — GradientBoosting; "
                   f"рядом в БД baseline-персистенс.")
    else:
        st.caption(f"Показаны все {gbr['station_code'].nunique()} городов. "
                   f"Выдели нужные на карте или в списке.")
    table = (gbr[["city", "target_date", "predicted_tmin", "predicted_tavg", "predicted_tmax"]]
             .rename(columns={"target_date": "дата", "predicted_tmin": "tmin, °C",
                              "predicted_tavg": "tavg, °C", "predicted_tmax": "tmax, °C"})
             .sort_values("city"))
    st.dataframe(table, width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.1f")
                                for c in ["tmin, °C", "tavg, °C", "tmax, °C"]})

# --- Точность прогноза (бэктест) ---
st.subheader("Точность прогноза (бэктест на истории)")
try:
    ev = load_eval()
except Exception:
    ev = pd.DataFrame()
if ev.empty:
    st.info("Оценка точности ещё не посчитана: `docker compose run --rm loader python -m weather.evaluate`")
else:
    if chosen:
        ev = ev[ev["station_code"].isin(chosen)]
    if ev.empty:
        st.info("Нет оценок для выбранных городов.")
    else:
        daily = ev.groupby(["eval_date", "model"])["abs_error"].mean().reset_index()
        fig_e = px.line(daily, x="eval_date", y="abs_error", color="model",
                        labels={"abs_error": "|ошибка| tavg, °C", "eval_date": "дата", "model": "модель"},
                        title="Средняя абсолютная ошибка прогноза tavg по дням")
        st.plotly_chart(fig_e, width="stretch")
        overall = ev.groupby("model")["abs_error"].mean()
        mae_g, mae_p = overall.get("gbr"), overall.get("persistence")
        k1, k2, k3 = st.columns(3)
        if mae_g is not None:
            k1.metric("MAE модели (GBR)", f"{mae_g:.2f} °C")
        if mae_p is not None:
            k2.metric("MAE персистенса", f"{mae_p:.2f} °C")
        if mae_g is not None and mae_p is not None:
            k3.metric("Модель vs персистенс", f"{mae_p - mae_g:+.2f} °C",
                      help="Плюс = модель точнее baseline")
        st.caption("Бэктест: модель училась на истории до окна и предсказывала каждый его день. "
                   "Фильтруется выбором городов на карте.")

# --- Город подробнее ---
st.subheader("Город подробнее")
city_options = stations["city"].tolist()
default_idx = 0
if chosen:
    sub = stations.loc[stations["code"] == chosen[0], "city"]
    if not sub.empty and sub.iloc[0] in city_options:
        default_idx = city_options.index(sub.iloc[0])
city = st.selectbox("Город", city_options, index=default_idx)
code = stations.loc[stations["city"] == city, "code"].iloc[0]
days = st.slider("Период, дней", 3, 60, 14)

series = load_series(code, days)
if series.empty:
    st.info("Нет наблюдений за выбранный период.")
else:
    cc1, cc2 = st.columns(2)
    cc1.plotly_chart(px.line(series, x="ts", y="temp_c", title="Температура, °C"),
                     width="stretch")
    cc2.plotly_chart(px.line(series, x="ts", y="pressure_mm", title="Давление, мм рт. ст."),
                     width="stretch")
    st.plotly_chart(px.line(series, x="ts", y="humidity_pct", title="Влажность, %"),
                    width="stretch")
    st.dataframe(series.tail(24), width="stretch")


# --- Конструктор модели ---
st.subheader("🛠 Конструктор модели")
st.caption("Сравни две модели бок о бок: выбери город, цель, признаки и **две модели** — "
           "график (факт / A / B) и прогнозы пересчитаются вживую.")
kc1, kc2, kc3 = st.columns(3)
con_city = kc1.selectbox("Город", city_options, index=default_idx, key="con_city")
con_code = stations.loc[stations["city"] == con_city, "code"].iloc[0]
con_target = kc2.selectbox("Цель", models.TARGETS, key="con_target")
mlist = list(models.MODELS)
mcol_a, mcol_b = st.columns(2)
model_a = mcol_a.selectbox("Модель A", mlist, index=0, key="con_model_a")
model_b = mcol_b.selectbox("Модель B", mlist, index=min(1, len(mlist) - 1), key="con_model_b")
con_feats = st.multiselect("Признаки в моделях", models.ALL_FEATURES,
                           default=models.ALL_FEATURES, key="con_feats")

if not con_feats:
    st.warning("Выбери хотя бы один признак.")
else:
    daily = load_daily_city(con_code)
    try:
        ra = models.backtest(daily, model_a, con_feats, con_target)
        rb = models.backtest(daily, model_b, con_feats, con_target)
    except Exception as e:
        st.error(f"Не удалось обучить: {e}")
        ra = rb = None
    if not ra or not rb:
        st.info("Мало данных по этому городу для обучения.")
    else:
        winner = model_a if ra["mae_model"] <= rb["mae_model"] else model_b
        q1, q2, q3, q4 = st.columns(4)
        q1.metric(f"MAE — {model_a}", f"{ra['mae_model']:.2f} °C")
        q2.metric(f"MAE — {model_b}", f"{rb['mae_model']:.2f} °C")
        q3.metric("персистенс (ref)", f"{ra['mae_persist']:.2f} °C")
        q4.metric(f"{model_a} − {model_b}", f"{ra['mae_model'] - rb['mae_model']:+.2f} °C",
                  help="Минус = модель A точнее B")
        plot = pd.DataFrame({"дата": ra["dates"], "факт": ra["actual"],
                             f"A · {model_a}": ra["pred"], f"B · {model_b}": rb["pred"]})
        long = plot.melt("дата", var_name="ряд", value_name="°C")
        figk = px.line(long, x="дата", y="°C", color="ряд",
                       title=f"{con_target} в «{con_city}»: факт vs {model_a} vs {model_b} "
                             f"(тест {ra['n_test']} дн.)")
        st.plotly_chart(figk, width="stretch")
        st.success(f"Прогноз **{con_target}** на {ra['tomorrow_date']}:  "
                   f"**{model_a}** → {ra['tomorrow']:.1f} °C   ·   "
                   f"**{model_b}** → {rb['tomorrow']:.1f} °C   ·   точнее здесь: **{winner}**")

        with st.expander("📊 Все модели разом (bar-чарт)"):
            cmp = models.compare_models(daily, con_feats, con_target)
            if cmp is not None:
                colors = {m: ("#9aa0a6" if m == "Персистенс" else "#4c8bf5") for m in cmp["model"]}
                fig_cmp = px.bar(cmp, x="model", y="MAE", color="model", color_discrete_map=colors,
                                 title=f"MAE моделей — {con_target} в «{con_city}» (меньше = точнее)",
                                 labels={"MAE": "MAE, °C", "model": "модель"})
                fig_cmp.update_layout(showlegend=False, height=360)
                st.plotly_chart(fig_cmp, width="stretch")
