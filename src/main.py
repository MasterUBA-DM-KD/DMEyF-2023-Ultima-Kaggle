import logging
import logging.config
import os

import duckdb

from src.constants import (
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_TRACKING_URI,
    PATH_CLASE_BINARIA,
    PATH_CLASE_TERNARIA,
    PATH_CRUDO,
    QUERY_DF_TEST,
    QUERY_DF_TRAIN,
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
    df_train = get_dataframe(con, QUERY_DF_TRAIN)
    df_test = get_dataframe(con, QUERY_DF_TEST)
    logger.info("Preprocess for training - Finished")

    logger.info("Closing connection to database")
    con.close()

    logger.info("Training - Started")
    model, run_name = training_loop(df_train)
    logger.info("Training - Finished")
