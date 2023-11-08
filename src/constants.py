# GENERAL PURPOSE
RANDOM_STATE = 42
SEEDS = [100057, 101183, 195581, 210913, 219761, 221243, 222199, 222217]

# ETL
RUN_ETL = False
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

# TRAINING
N_TRIALS_OPTIMIZE = 10
PRUNER_WARMUP_STEPS = 5
EARLY_STOPPING_ROUNDS = 5
COLS_TO_DROP = ["clase_binaria", "foto_mes", "numero_de_cliente"]
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
]
VALIDATION_MONTHS = [202107, 202108]
TEST_MONTH = [202109]
BASE_PATH_PREDICTIONS = "../buckets/b1/datasets/processed/predictions"

in_clause_training = ", ".join([str(i) for i in TRAINING_MONTHS])
QUERY_DF_TRAIN = f"SELECT * FROM competencia_03 WHERE foto_mes IN ({in_clause_training})"

in_clause_validation = ", ".join([str(i) for i in VALIDATION_MONTHS])
QUERY_DF_VALID = f"SELECT * FROM competencia_03 WHERE foto_mes IN ({in_clause_validation})"

in_clause_test = ", ".join([str(i) for i in TEST_MONTH])
QUERY_DF_TEST = f"SELECT * FROM competencia_03 WHERE foto_mes IN ({in_clause_test})"


PARAMS_LGB = {
    "metric": "auc",
    "objective": "binary",
    "boosting_type": "gbdt",
    "force_col_wise": True,
    "feature_pre_filter": False,
    "verbosity": -1,
    "seed": RANDOM_STATE,
    "n_jobs": -1,
    "bagging_fraction": 0.1,
    "bagging_freq": 2,
    "feature_fraction": 0.5,
    "lambda_l1": 0.0035621591425357845,
    "lambda_l2": 0.1354162723749174,
    "learning_rate": 0.1497926694870291,
    "max_depth": 10,
    "min_data_in_leaf": 6600,
    "min_gain_to_split": 3.4328502146787363,
    "num_leaves": 11,
}
