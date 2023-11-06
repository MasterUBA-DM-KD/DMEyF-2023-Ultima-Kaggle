import logging
import os

import lightgbm as lgb
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from optuna.integration import MLflowCallback
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score

from src.constants import EVALUATOR_CONFIG, RANDOM_STATE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mlflc = MLflowCallback(
    tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
    metric_name="f-score",
    create_experiment=False,
    mlflow_kwargs={
        "nested": True,
    },
)


def objective(trial: optuna.Trial, dtrain: lgb.Dataset, dvalid: lgb.Dataset, X_test, y_test):
    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": 1,
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(
        param,
        dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            optuna.integration.LightGBMPruningCallback(trial, "auc", "valid"),
            lgb.early_stopping(stopping_rounds=5, verbose=False),
        ],
    )

    preds = gbm.predict(X_test)
    preds = np.rint(preds)
    f_score = f1_score(y_test, preds)

    return f_score


def training_loop(df_train: pd.DataFrame, df_valid: pd.DataFrame) -> float:
    logger.info("Starting training loop")
    mlflow.lightgbm.autolog()

    X_train = df_train.drop(columns=["clase_binaria"], axis=1)
    y_train = df_train["clase_binaria"]

    X_test = df_valid.drop(columns=["clase_binaria"], axis=1)
    y_test = df_valid["clase_binaria"]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test)

    sampler = TPESampler(seed=RANDOM_STATE)

    logger.info("MLFlow Run - Started")
    with mlflow.start_run():
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize", sampler=sampler
        )
        study.optimize(
            lambda trial: objective(trial, dtrain, dvalid, X_test.values, y_test.values),
            n_trials=10,
            n_jobs=3,
            callbacks=[mlflc],
        )

        model = lgb.LGBMClassifier(**study.best_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        preds = np.rint(preds)
        f_score = f1_score(y_test, preds)
        mlflow.log_metric("f-score", f_score)

        model_info = mlflow.lightgbm.log_model(model, "model")

        eval_data = X_test.copy()
        eval_data["target"] = y_test.copy()

        logger.info("MLFlow evaluation - Started")
        mlflow.evaluate(
            model_info.model_uri,
            data=eval_data,
            targets="target",
            model_type="classifier",
            evaluators="default",
            evaluator_config=EVALUATOR_CONFIG,
        )
        logger.info("MLFlow evaluation - Finished")
    logger.info("MLFlow Run - Finished")
