import logging

import lightgbm
import optuna.integration.lightgbm as lgb
import pandas as pd
from lightgbm import log_evaluation

from src.constants import COLS_TO_DROP, NFOLD, PARAMS_LGBM, RANDOM_STATE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def training_loop(df_train: pd.DataFrame) -> None:
    logger.info("Starting training loop")

    X_train = df_train.drop(columns=COLS_TO_DROP, axis=1).copy()

    # y_train_ternaria = df_train["clase_ternaria"].copy()
    # y_train_ternaria = y_train_ternaria.map({"CONTINUA": 0, "BAJA+1": 1, "BAJA+2": 2})
    y_train_binaria = df_train["clase_binaria"].copy()

    dtrain = lightgbm.Dataset(X_train, label=y_train_binaria, free_raw_data=False)

    tuner = lgb.LightGBMTunerCV(
        PARAMS_LGBM,
        dtrain,
        nfold=NFOLD,
        stratified=True,
        callbacks=[log_evaluation(100)],
        seed=RANDOM_STATE,
        optuna_seed=RANDOM_STATE,
    )

    tuner.run()

    logger.info("Best score:", tuner.best_score)
    best_params = tuner.best_params
    logger.info("Best params:", best_params)

    logger.info("Params:")
    for key, value in best_params.items():
        logger.info("  {}: {}".format(key, value))

    # Train model with best parameters

    logger.info("Training model with best parameters")

    model = lightgbm.train(
        best_params,
        dtrain,
        callbacks=[log_evaluation(100)],
    )

    logger.info("Saving model")

    model.save_model("model.txt")

    logger.info("Training loop finished")

    # predict with model

    logger.info("Predicting with model")
