import logging
import logging.config

import duckdb

from src.constants import PATH_CLASE_BINARIA, PATH_CRUDO, RUN_ETL
from src.model.training import training_loop
from src.preprocess.etl import extract, preprocess_training, transform

logging.config.fileConfig(fname="~/buckets/b1/logs/run.log", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Connecting to in-memory database")
    con = duckdb.connect(database=":memory:", read_only=False)
    if RUN_ETL:
        logger.warning("Running the whole ETL")

        logger.info("Extract - Started")
        extract(con, PATH_CRUDO)
        logger.info("Extract - Finished")

        logger.info("Transform - Started")
        transform(con)
        logger.info("Transform - Finished")
    else:
        logger.warning("Reading from %s - Transform will be skipped", PATH_CLASE_BINARIA)

        logger.info("Extract - Started")
        extract(con, PATH_CLASE_BINARIA)
        logger.info("Extract - Finished")

    logger.info("Preprocess for training - Started")
    df_train, df_valid, df_test = preprocess_training(con)
    logger.info("Preprocess for training - Finished")

    logger.info("Closing connection to in-memory database")
    con.close()

    logger.info("Training - started")
    training_loop(df_train, df_valid)
    logger.info("Training - Finished")
