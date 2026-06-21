-- Схема БД проекта «Погода России».
-- Применяется автоматически при первом старте контейнера postgres
-- (db/ монтируется в /docker-entrypoint-initdb.d).

CREATE TABLE IF NOT EXISTS stations (
    code       TEXT PRIMARY KEY,        -- код станции в URL meteo9 (напр. Se7X)
    city       TEXT NOT NULL,
    region     TEXT,
    slug       TEXT,
    lat        DOUBLE PRECISION,
    lon        DOUBLE PRECISION,
    tz_offset  INTEGER                  -- сдвиг местного времени от UTC, часы
);

CREATE TABLE IF NOT EXISTS observations (
    station_code    TEXT NOT NULL REFERENCES stations(code) ON DELETE CASCADE,
    ts              TIMESTAMP NOT NULL,   -- момент наблюдения (как отдаёт источник)
    temp_c          REAL,
    pressure_mm     REAL,
    humidity_pct    SMALLINT,
    cloud_pct       REAL,
    wind_dir        TEXT,
    wind_dir_code   SMALLINT,
    wind_speed_min  REAL,
    wind_speed_max  REAL,
    precip_mm       REAL,
    weather_code    SMALLINT,
    loaded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- ключ делает повторную загрузку идемпотентной: один замер = одна строка
    PRIMARY KEY (station_code, ts)
);

CREATE INDEX IF NOT EXISTS idx_observations_ts ON observations (ts);
CREATE INDEX IF NOT EXISTS idx_observations_station_ts ON observations (station_code, ts);

CREATE TABLE IF NOT EXISTS forecasts (
    station_code    TEXT NOT NULL REFERENCES stations(code) ON DELETE CASCADE,
    target_date     DATE NOT NULL,        -- на какой день прогноз
    predicted_tmin  REAL,
    predicted_tmax  REAL,
    predicted_tavg  REAL,
    model           TEXT NOT NULL,        -- имя модели (напр. 'gbr', 'persistence')
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (station_code, target_date, model)
);

CREATE TABLE IF NOT EXISTS forecast_eval (
    station_code  TEXT NOT NULL REFERENCES stations(code) ON DELETE CASCADE,
    eval_date     DATE NOT NULL,        -- день, на который делался прогноз (факт уже известен)
    model         TEXT NOT NULL,        -- 'gbr' | 'persistence'
    predicted     REAL,
    actual        REAL,
    abs_error     REAL,
    PRIMARY KEY (station_code, eval_date, model)
);
