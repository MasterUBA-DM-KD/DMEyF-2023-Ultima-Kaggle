import logging
import os

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMClassifier

from src.constants import BASE_PATH_PREDICTIONS, COLS_TO_DROP, EARLY_STOPPING_ROUNDS, SEEDS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

early_stopper = lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True)


def predictions_per_seed(
    df_train: pd.DataFrame, df_valid: pd.DataFrame, df_test: pd.DataFrame, model: LGBMClassifier, run_name: str
) -> None:
    logger.info("Starting predictions per seed")
    base_path = os.path.join(BASE_PATH_PREDICTIONS, run_name)
    os.makedirs(base_path, exist_ok=True)

    X_train = df_train.drop(columns=COLS_TO_DROP, axis=1).copy()
    X_valid = df_valid.drop(columns=COLS_TO_DROP, axis=1).copy()
    X_test = df_test.drop(columns=COLS_TO_DROP, axis=1).copy()

    y_train = df_train["clase_binaria"]
    y_valid = df_valid["clase_binaria"]

    final_preds = df_test["numero_de_cliente"].to_frame()

    params = model.get_params()
    params["objective"] = "binary"
    params["metric"] = "auc"
    params["force_col_wise"] = True
    params["n_jobs"] = -1

    for seed in SEEDS:
        logger.info("Training with seed %s", seed)
        params["random_state"] = seed
        gbm = lgb.LGBMClassifier(**params)
        gbm.fit(
            X=X_train,
            y=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_names=["train", "valid"],
            eval_metric="auc",
            init_model=model,
            callbacks=[early_stopper],
        )

        preds = gbm.predict_proba(X_test, num_threads=10)
        final_preds[f"seed_{seed}"] = preds[:, 1]

    final_preds["Predicted"] = final_preds.iloc[:, 1:].mean(axis=1)

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
