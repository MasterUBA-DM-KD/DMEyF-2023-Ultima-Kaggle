import logging
import os

import pandas as pd
from lightgbm import Booster

from src.model.training import training_loop

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///database/mlruns.db"
os.environ["MLFLOW_ARTIFACT_ROOT"] = "mlruns"

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_training():
    df_train = pd.read_csv("datasets/processed/data_for_test.csv")
    model, run_name = training_loop(df_train)

    assert isinstance(model, Booster)
