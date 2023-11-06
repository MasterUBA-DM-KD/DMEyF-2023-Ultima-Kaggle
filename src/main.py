from datetime import datetime

import duckdb

from src.model.training import training_loop
from src.preprocess.etl import extract, preprocess_training, transform

if __name__ == "__main__":
    con = duckdb.connect(database=":memory:", read_only=False)
    print("Extract", datetime.now())
    extract(con)
    print("Transform", datetime.now())
    transform(con)

    print("Preprocess - Training", datetime.now())
    df_train, df_valid, df_test = preprocess_training(con)

    print("Training", datetime.now())
    training_loop(df_train, df_valid)
