import logging
import logging.config

import duckdb

from src.constants import PATH_CLASE_BINARIA, PATH_CRUDO, RUN_ETL, TRAINING_MONTHS, VALIDATION_MONTHS
from src.model.training import training_loop

# from src.model.inference import predictions_per_seed
from src.preprocess.etl import extract, get_dataframe, transform

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logger.info("Connecting to in-memory database")
    con = duckdb.connect(database=":memory:", read_only=False)

    if RUN_ETL:
        logger.warning("Running the whole ETL")

        logger.info("Extract - Started")
        extract(con, PATH_CRUDO)
        logger.info("Extract - Finished")

        logger.info("Transform - Started")
        transform(con, True, True)
        logger.info("Transform - Finished")
    else:
        logger.warning("Reading from %s - Transform will be skipped", PATH_CLASE_BINARIA)

        logger.info("Extract - Started")
        extract(con, PATH_CLASE_BINARIA)
        logger.info("Extract - Finished")

    logger.info("Preprocess for training - Started")
    in_clause_training = ", ".join([str(i) for i in TRAINING_MONTHS])
    query = f"SELECT * FROM competencia_03 WHERE foto_mes IN ({in_clause_training})"
    df_train = get_dataframe(con, query)

    in_clause_validation = ", ".join([str(i) for i in VALIDATION_MONTHS])
    query = f"SELECT * FROM competencia_03 WHERE foto_mes IN ({in_clause_validation})"
    df_valid = get_dataframe(con, query)

    # in_clause_test = ", ".join([str(i) for i in TEST_MONTH])
    # query = f"SELECT * FROM competencia_03 WHERE foto_mes IN ({in_clause_test})"
    # df_test = get_dataframe(con, query)
    logger.info("Preprocess for training - Finished")

    logger.info("Closing connection to in-memory database")
    con.close()

    logger.info("Training - started")
    model, run_name = training_loop(df_train, df_valid)
    logger.info("Training - Finished")

    # logger.info("Inference - started")
    # predictions_per_seed(df_test, model)
    # logger.info("Inference - Finished")
