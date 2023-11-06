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


def objective(trial: optuna.Trial, dtrain: lgb.Dataset, dvalid: lgb.Dataset, X_test, y_test, X_train, y_train):
    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": 1,
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        "force_col_wise": True,
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1),
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
    WEIGHTS = y_train.value_counts(normalize=True).min() / y_train.value_counts(normalize=True)
    TRAIN_WEIGHTS = (
        pd.DataFrame(y_train.rename("old_target"))
        .merge(WEIGHTS, how="left", left_on="old_target", right_on=WEIGHTS.index)
        .values
    )

    X_test = df_valid.drop(columns=["clase_binaria"], axis=1)
    y_test = df_valid["clase_binaria"]

    dtrain = lgb.Dataset(X_train, label=y_train, weight=TRAIN_WEIGHTS[:, 1])
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    sampler = TPESampler(seed=RANDOM_STATE)

    logger.info("MLFlow Run - Started")
    with mlflow.start_run():
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize", sampler=sampler
        )
        study.optimize(
            lambda trial: objective(
                trial, dtrain, dvalid, X_test.values, y_test.values, X_train.values, y_train.values
            ),
            n_trials=10,
            n_jobs=2,
            callbacks=[mlflc],
            gc_after_trial=True,
        )

        best_model = lgb.LGBMClassifier(**study.best_params, random_state=RANDOM_STATE)
        best_model = best_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        preds = best_model.predict(X_test)
        preds = np.rint(preds)
        f_score = f1_score(y_test, preds)
        mlflow.log_metric("f-score", f_score)

        model_info = mlflow.lightgbm.log_model(best_model, "model")

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

    return best_model
