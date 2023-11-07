import logging
import os

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier

from src.constants import SEEDS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predictions_per_seed(df_test: pd.DataFrame, model: LGBMClassifier, run_name: str) -> None:
    logger.info("Starting predictions per seed")
    base_path = f"datasets/processed/predictions/{run_name}"
    os.makedirs(base_path, exist_ok=True)

    X_test = df_test.drop(columns=["clase_binaria", "foto_mes", "numero_de_cliente"], axis=1)
    final_preds = df_test["numero_de_cliente"].to_frame()

    for seed in SEEDS:
        logger.info("Predicting for seed %s", seed)
        params = model.get_params()
        params["random_state"] = seed
        gbm = lgb.train(
            params,
            init_model=model,
        )
        preds = gbm.predict_proba(X_test)
        final_preds[f"seed_{seed}"] = preds[:, 1]

    final_preds["prediction"] = final_preds.iloc[:, 1:].mean(axis=1)
    final_preds = final_preds[["numero_de_cliente", "prediction"]]
    final_preds = final_preds.sort_values(by="prediction", ascending=False)
    final_preds = final_preds.reset_index(drop=True)

    for cut in range(5000, 20000, 500):
        final_preds.loc[0:cut, "prediction"] = 1
        final_preds.loc[cut:, "prediction"] = 0
        final_preds.to_csv(f"datasets/processed/predictions/{run_name}_{cut}_envios.csv", index=False)
