import logging
import logging.config
import os

import duckdb

from src.constants import MLFLOW_TRACKING_URI, PATH_CRUDO, PATH_FINAL_PARQUET, QUERY_DF_TEST, QUERY_DF_TRAIN, RUN_ETL
from src.model.inference import predictions_per_seed
from src.model.training import training_loop
from src.preprocess.etl import extract, load, transform
from src.preprocess.utils import get_dataframe

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

if __name__ == "__main__":
    logger.info("Connecting to in-memory database")
    con = duckdb.connect(database=":memory:", read_only=False)

    if RUN_ETL:
        logger.warning("Running the whole ETL")

        logger.info("Extract - Started")
        extract(con, PATH_CRUDO)
        logger.info("Extract - Finished")

        logger.info("Transform - Started")
        transform(con, True, True, True, True, True)
        logger.info("Transform - Finished")

        logger.info("Load - Started")
        load(con, PATH_FINAL_PARQUET)
        logger.info("Load - Finished")
    else:
        logger.warning("Reading from %s - Transform will be skipped", PATH_FINAL_PARQUET)

        logger.info("Extract - Started")
        extract(con, PATH_FINAL_PARQUET)
        logger.info("Extract - Finished")

    logger.info("Preprocess for training - Started")
    df_train = get_dataframe(con, QUERY_DF_TRAIN)
    df_test = get_dataframe(con, QUERY_DF_TEST)

    logger.info("Closing connection to database")
    con.close()

    df_train["clase_binaria"] = df_train["clase_ternaria"].map({"BAJA+2": 2, "BAJA+1": 1, "CONTINUA": 0})
    logger.info("Preprocess for training - Finished")
    logger.info("Training - Started")
    best_model, run_name = training_loop(df_train)
    predictions_per_seed(df_train, best_model, run_name)
    logger.info("Training - Finished")
