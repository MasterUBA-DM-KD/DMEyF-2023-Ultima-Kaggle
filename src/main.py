import logging
import logging.config
import os
import re

import duckdb
from sklearn.model_selection import train_test_split

from src.constants import (
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_TRACKING_URI,
    PATH_CLASE_BINARIA,
    PATH_CLASE_TERNARIA,
    PATH_CRUDO,
    QUERY_DF_TRAIN,
    RANDOM_STATE,
    RUN_ETL,
)
from src.model.training import training_loop
from src.preprocess.etl import extract, get_dataframe, transform

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
os.environ["MLFLOW_ARTIFACT_ROOT"] = MLFLOW_ARTIFACT_ROOT

if __name__ == "__main__":
    logger.info("Connecting to in-memory database")
    con = duckdb.connect(database=":memory:", read_only=False)

    if RUN_ETL:
        logger.warning("Running the whole ETL")

        logger.info("Extract - Started")
        extract(con, PATH_CRUDO)
        logger.info("Extract - Finished")

        logger.info("Transform - Started")
        transform(con, PATH_CLASE_TERNARIA, PATH_CLASE_BINARIA, False, True)
        logger.info("Transform - Finished")
    else:
        logger.warning("Reading from %s - Transform will be skipped", PATH_CLASE_BINARIA)

        logger.info("Extract - Started")
        extract(con, PATH_CLASE_BINARIA)
        logger.info("Extract - Finished")

    logger.info("Preprocess for training - Started")
    df_full = get_dataframe(con, QUERY_DF_TRAIN)

    logger.info("Closing connection to database")
    con.close()

    logger.info("Cleaning column names")
    df_full = df_full.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

    logger.info("Splitting into train and validation")
    df_full["stratify"] = df_full["clase_ternaria"].astype(str) + df_full["foto_mes"].astype(str)
    df_train, df_valid = train_test_split(
        df_full, test_size=0.05, random_state=RANDOM_STATE, stratify=df_full["stratify"]
    )

    df_train = df_train.drop(columns=["stratify"], axis=1)
    df_valid = df_valid.drop(columns=["stratify"], axis=1)
    logger.info("Preprocess for training - Finished")

    logger.info("Training - Started")
    model, run_name = training_loop(df_train, df_valid)
    logger.info("Training - Finished")
