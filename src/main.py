# docformatter Package Imports
from src.model.training import training_loop
from src.preprocess.etl import extract, load, transform

if __name__ == "__main__":
    X, y = extract()
    X, y = transform(X, y)
    X, y = load(X, y)

    training_loop(X, y)
