import logging
import os

import pandas as pd
from lightgbm import Booster

from src.constants import BASE_PATH_PREDICTIONS, COLS_TO_DROP, SEEDS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predictions_per_seed(df_test: pd.DataFrame, model: Booster, run_name: str) -> None:
    logger.info("Starting predictions per seed")

    base_path = os.path.join(BASE_PATH_PREDICTIONS, run_name)
    os.makedirs(base_path, exist_ok=True)

    X_test = df_test.drop(columns=COLS_TO_DROP, axis=1).copy()
    final_preds = df_test["numero_de_cliente"].to_frame()

    for seed in SEEDS:
        logger.info("Training with seed %s", seed)
        model.params["random_state"] = seed

        logger.info("Prediction with seed %s", seed)
        preds = model.predict(X_test)
        final_preds[f"seed_{seed}"] = preds

    final_preds["Predicted"] = final_preds.iloc[:, 1:].mean(axis=1)

    logger.info("Aggregating predictions - all seeds")
    final_preds = final_preds.sort_values(by="Predicted", ascending=False)
    final_preds = final_preds.reset_index(drop=True)
    final_preds.to_csv(os.path.join(base_path, "predictions.csv"), index=False)

    logger.info("Saving aggregated predictions")
    for cut in range(5000, 20000, 500):
        final_preds_cut = final_preds.copy()
        final_preds_cut = final_preds_cut[["numero_de_cliente", "Predicted"]]
        final_preds_cut.loc[0:cut, "Predicted"] = True
        final_preds_cut.loc[cut:, "Predicted"] = False
        out_path = os.path.join(base_path, f"{cut}.csv")
        final_preds_cut.to_csv(out_path, index=False)

    logger.info("Saved predictions to %s", base_path)
    logger.info("Finished predictions per seed")
