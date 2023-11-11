# GENERAL PURPOSE
RUN_ETL = False
RANDOM_STATE = 42
RANDOM_STATE_EXTRA = 101
MLFLOW_TRACKING_URI = "sqlite:///database/mlruns.db"
MLFLOW_ARTIFACT_ROOT = "gs://mlflow-artifacts-uribe/mlruns"
WEIGHTS_TRAINING = {"BAJA+2": 1.0000002, "BAJA+1": 1.0000001, "CONTINUA": 1.0}
MATRIX_GANANCIA = {1.0000002: 273000, 1.0000001: -7000, 1.0: -7000}
SEEDS = [100057, 101183, 195581, 210913, 219761, 221243, 222199, 222217]

# ETL
PATH_FINAL_CSV = "~/buckets/b1/datasets/processed/competencia_03.csv"
PATH_FINAL_PARQUET = "~/buckets/b1/datasets/processed/competencia_03.parquet"
PATH_CRUDO = "~/buckets/b1/datasets/interim/competencia_03_crudo.parquet"
PATH_CLASE_TERNARIA = "~/buckets/b1/datasets/processed/competencia_03_clase_ternaria.parquet"
PATH_CLASE_BINARIA = "~/buckets/b1/datasets/processed/competencia_03_clase_binaria.parquet"

PATH_SMALL = "datasets/raw/competencia_03_small.parquet"
PATH_CLASE_TERNARIA_SMALL = "datasets/processed/competencia_03_clase_ternaria_small.parquet"
PATH_CLASE_BINARIA_SMALL = "datasets/processed/competencia_03_clase_binaria_small.parquet"

URL_INFLATION = "https://www.indec.gob.ar/ftp/cuadros/economia/sh_ipc_08_23.xls"
PATH_INFLATION_RAW = "~/buckets/b1/datasets/raw/inflation.parquet"
PATH_INFLATION_FINAL = "~/buckets/b1/datasets/processed/inflation.parquet"

INFLATION_FILE = "sql/inflation.sql"

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

TEND_FILES = [
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
PRUNER_WARMUP_STEPS = 5
COLS_TO_DROP = ["clase_ternaria", "clase_binaria"]
EVALUATOR_CONFIG = {"explainability_algorithm": "permutation", "metric_prefix": "evaluation_"}

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

TEST_MONTH = [202109]
BASE_PATH_PREDICTIONS = "../buckets/b1/datasets/processed/predictions"

in_clause_training = ", ".join([str(i) for i in TRAINING_MONTHS])
QUERY_DF_TRAIN = f"SELECT * FROM competencia_03 WHERE foto_mes IN ({in_clause_training})"

in_clause_test = ", ".join([str(i) for i in TEST_MONTH])
QUERY_DF_TEST = f"SELECT * FROM competencia_03 WHERE foto_mes IN ({in_clause_test})"


PARAMS_LGBM = {
    "boosting": "gbdt",
    "objective": "binary",
    "metric": "custom",
    "first_metric_only": True,
    "boost_from_average": True,
    "feature_pre_filter": False,
    "force_row_wise": True,
    "verbosity": -1,
    "min_gain_to_split": 0.0,
    "min_sum_hessian_in_leaf": 0.001,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "max_bin": 31,
    "bagging_fraction": 1.0,
    "pos_bagging_fraction": 1.0,
    "neg_bagging_fraction": 1.0,
    "is_unbalance": False,
    "scale_pos_weight": 1.0,
    "drop_rate": 0.1,
    "max_drop": 50,
    "skip_drop": 0.5,
    "extra_trees": True,
}
