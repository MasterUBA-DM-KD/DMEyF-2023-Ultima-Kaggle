import logging
import os
from typing import Tuple

import lightgbm
import numpy as np
import optuna.integration.lightgbm as lgb
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from optuna.integration import MLflowCallback

from src.constants import COLS_TO_DROP, MATRIX_GANANCIA, NFOLD, PARAMS_LGBM, RANDOM_STATE, WEIGHTS_TRAINING

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calculate_ganancia(preds: np.ndarray, data: lightgbm.Dataset) -> Tuple[str, float, bool]:
    metric_name = "ganancia"
    is_higher_better = True

    label = data.get_label()
    weights = data.get_weight()

    ganancia = pd.DataFrame({"preds": preds, "label": label, "weights": weights})
    ganancia["preds"] = np.rint(preds)
    ganancia["costo"] = ganancia["weights"].map(MATRIX_GANANCIA)
    ganancia["ganancia"] = ganancia["preds"] * ganancia["costo"]
    ganancia_total = float(ganancia["ganancia"].sum())

    return metric_name, -1.0 * ganancia_total, is_higher_better


def training_loop(df_train: pd.DataFrame) -> None:
    logger.info("Starting training loop")
    mlflow_callback = MLflowCallback(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        metric_name="ganancia",
        create_experiment=False,
        mlflow_kwargs={
            "nested": True,
        },
    )

    X_train = df_train.drop(columns=COLS_TO_DROP, axis=1).copy()

    y_train_ternaria = df_train["clase_ternaria"].copy()
    y_train_binaria = df_train["clase_binaria"].copy()

    train_weights = y_train_ternaria.to_frame()
    train_weights["weights"] = train_weights["clase_ternaria"].map(WEIGHTS_TRAINING)

    dtrain = lightgbm.Dataset(X_train, label=y_train_binaria, weight=train_weights["weights"], free_raw_data=False)

    tuner = lgb.LightGBMTunerCV(
        PARAMS_LGBM,
        dtrain,
        nfold=NFOLD,
        stratified=True,
        callbacks=[early_stopping(100), log_evaluation(100)],
        seed=RANDOM_STATE,
        feval=calculate_ganancia,
        optuna_seed=RANDOM_STATE,
        optuna_callbacks=[mlflow_callback],
    )

    tuner.run()

    logger.info("Best score:", tuner.best_score)
    best_params = tuner.best_params
    logger.info("Best params:", best_params)

    logger.info("Params:")
    for key, value in best_params.items():
        logger.info("  {}: {}".format(key, value))
