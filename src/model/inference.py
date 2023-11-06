import pandas as pd
from lightgbm import LGBMClassifier

from src.constants import SEEDS


def preds_per_seed(df_test: pd.DataFrame, model: LGBMClassifier, run_name: str) -> None:
    for seed in SEEDS:
        model.random_state = seed
        preds = model.predict(df_test.drop(columns=["clase_binaria"], axis=1))
        preds = pd.DataFrame(preds, columns=["preds"])
        preds.to_csv(f"predictions/preds_{seed}.csv", index=False)
