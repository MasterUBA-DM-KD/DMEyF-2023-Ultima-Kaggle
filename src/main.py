import duckdb

from src.constants import DATABASE_PATH
from src.model.training import training_loop
from src.preprocess.etl import extract, preprocess_training, transform

if __name__ == "__main__":
    con = duckdb.connect(database=DATABASE_PATH, read_only=False)
    extract(con)
    transform(con)

    df_train, df_valid, df_test = preprocess_training(con)

    training_loop(df_train, df_valid)
