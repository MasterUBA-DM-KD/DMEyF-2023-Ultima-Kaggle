from datetime import datetime

import duckdb

# from src.model.training import training_loop
from src.preprocess.etl import extract, preprocess_training, transform

if __name__ == "__main__":
    con = duckdb.connect(database=":memory:", read_only=False)
    print("Extract", datetime.now())
    extract(con)
    print("Transform", datetime.now())
    transform(con)

    print("Preprocess", datetime.now())
    preprocess_training(con)

    # training_loop(df_train, df_valid)
