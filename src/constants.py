RUN_ETL = False

RANDOM_STATE = 42
GANANCIA_METRIC = True

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
]
VALIDATION_MONTHS = [202107, 202108]
TEST_MONTH = [202109]

DATABASE_PATH = "database/competencia_03.duckdb"

PATH_CRUDO = "~/buckets/b1/datasets/raw/competencia_03_crudo.parquet"
PATH_CLASE_TERNARIA = "~/buckets/b1/datasets/processed/competencia_03_clase_ternaria.parquet"
PATH_CLASE_BINARIA = "~/buckets/b1/datasets/processed/competencia_03_clase_binaria.parquet"

PATH_SMALL = "~/buckets/b1/datasets/raw/competencia_03_small.parquet"
PATH_CLASE_TERNARIA_CSV_SMALL = "~/buckets/b1/datasets/processed/competencia_03_clase_ternaria_small.csv"

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
SEEDS = [100057, 101183, 195581, 210913, 219761, 221243, 222199, 222217, 222221, 222223, 222227, 222229]
