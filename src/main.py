import logging
import logging.config

import duckdb

from src.constants import DATABASE_PATH, PATH_CRUDO, RUN_ETL
from src.model.training import training_loop
from src.preprocess.etl import extract, preprocess_training, transform

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logger.info("Connecting to in-memory database")
    con = duckdb.connect(database=DATABASE_PATH, read_only=False)

    if RUN_ETL:
        logger.warning("Running the whole ETL")

        logger.info("Extract - Started")
        extract(con, PATH_CRUDO)
        logger.info("Extract - Finished")

        logger.info("Transform - Started")
        transform(con)
        logger.info("Transform - Finished")
    else:
        logger.warning("Reading from %s - Transform will be skipped", DATABASE_PATH)

    logger.info("Preprocess for training - Started")
    df_train, df_valid, df_test = preprocess_training(con)
    logger.info("Preprocess for training - Finished")

    logger.info("Closing connection to in-memory database")
    con.close()

    logger.info("Training - started")
    training_loop(df_train, df_valid)
    logger.info("Training - Finished")
