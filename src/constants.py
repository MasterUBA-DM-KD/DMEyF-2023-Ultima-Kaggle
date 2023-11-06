RANDOM_STATE = 42

TRAINING_MONTHS = [
    201901,
    201902,
    201903,
    201904,
    201905,
    201906,
    201907,
    201908,
    201908,
    201909,
    201910,
    201911,
    201912,
    202001,
    202002,
    202010,
    202011,
    202012,
    202101,
    202102,
    202103,
    202104,
    202105,
    202106,
    202107,
]
VALIDATION_MONTHS = [202108]
TEST_MONTH = [202109]

DATABASE_PATH = "database/competencia_03.duckdb"

PATH_CRUDO = "datasets/raw/competencia_03_crudo.parquet"
PATH_CLASE_TERNARIA = "datasets/processed/competencia_03_clase_ternaria.parquet"

PATH_SMALL = "datasets/raw/competencia_03_small.parquet"
PATH_CLASE_TERNARIA_CSV_SMALL = "datasets/processed/competencia_03_clase_ternaria_small.csv"

LAG_FILES = [
    "sql/lags_1.sql",
    "sql/lags_3.sql",
    "sql/lags_6.sql",
]

DELTA_FILES = [
    "sql/delta_lags_1.sql",
    "sql/delta_lags_3.sql",
    "sql/delta_lags_6.sql",
]

EVALUATOR_CONFIG = {"explainability_algorithm": "permutation", "metric_prefix": "evaluation_"}
