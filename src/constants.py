# GENERAL PURPOSE
RUN_ETL = False
RANDOM_STATE = 42
RANDOM_STATE_EXTRA = 101
SEEDS = [100057, 101183, 195581, 210913, 219761, 221243, 222199, 222217]

# MLFLOW
MLFLOW_TRACKING_URI = "http://34.122.201.241:5000/"

# PATHS
PATH_CRUDO = "~/buckets/b1/datasets/raw/competencia_03_crudo.parquet"
PATH_FINAL_PARQUET = "~/buckets/b1/datasets/processed/competencia_03.parquet"
DATABASE_PATH = "database/competencia_03.db"
OPTUNA_STORAGE = "sqlite:///database/optuna.db"

PATH_INFLATION_RAW = "~/buckets/b1/datasets/raw/inflation.parquet"
PATH_INFLATION_FINAL = "~/buckets/b1/datasets/processed/inflation.parquet"
URL_INFLATION = "https://www.indec.gob.ar/ftp/cuadros/economia/sh_ipc_08_23.xls"

BASE_PATH_PREDICTIONS = "datasets/processed/predictions"

INFLATION = "sql/inflation.sql"

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

TREND_FILES = [
    "sql/tend_3.sql",
    "sql/tend_6.sql",
]

MOVING_AVG_FILES = [
    "sql/moving_avg_3.sql",
    "sql/moving_avg_6.sql",
]

# TRAINING
NFOLD = 5
METRIC = "ganancia"
N_TRIALS_OPTIMIZE = 10
PRUNER_WARMUP_STEPS = 3
COLS_TO_DROP = ["clase_ternaria", "clase_binaria", "foto_mes", "clase_ternaria:1"]
COST_ENVIO = {1.0000002: 273000, 1.0000001: -7000, 1.0: -7000}
WEIGHTS_TRAINING = {"BAJA+2": 1.0000002, "BAJA+1": 1.0000001, "CONTINUA": 1.0}

TRAINING_MONTHS = [
    # 201901,
    # 201902,
    # 201903,
    # 201904,
    # 201905,
    # 201906,
    # 201907,
    # 201908,
    # 201908,
    # 201909,
    # 201910,
    # 201911,
    # 201912,
    202001,
    202002,
    202003,
    202004,
    202005,
    202006,
    202007,
    202008,
    202009,
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

TEST_MONTH = [202109]

in_clause_training = ", ".join([str(i) for i in TRAINING_MONTHS])
QUERY_DF_TRAIN = f"SELECT * FROM competencia_03 WHERE foto_mes IN ({in_clause_training})"

in_clause_test = ", ".join([str(i) for i in TEST_MONTH])
QUERY_DF_TEST = f"SELECT * FROM competencia_03 WHERE foto_mes IN ({in_clause_test})"
