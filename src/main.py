from src.model.training import preprocess, training_loop
from src.preprocess.etl import extract_transform_load

if __name__ == "__main__":
    extract_transform_load()
    X, y = preprocess()

    training_loop(X, y)
