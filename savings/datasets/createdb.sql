CREATE TABLE public.turnover (
  "stag"               NUMERIC(18,4),
  "age"                NUMERIC(18,4),
  "extraversion"       NUMERIC(18,4),
  "independ"           NUMERIC(18,4),
  "selfcontrol"        NUMERIC(18,4),
  "anxiety"            NUMERIC(18,4),
  "novator"            NUMERIC(18,4),
  "event"              NUMERIC(18,4)
);

COPY public.turnover(
  "stag",
  "age",
  "extraversion",
  "independ",
  "selfcontrol",
  "anxiety",
  "novator",
  "event"
) FROM '/var/lib/postgresql/savings/datasets/turnover.csv' DELIMITER ',' CSV HEADER;

CREATE TABLE public.models
(
    "model_id"         TEXT PRIMARY KEY,
    "model_name"       TEXT    NOT NULL,
    "model_params"     TEXT,
);
