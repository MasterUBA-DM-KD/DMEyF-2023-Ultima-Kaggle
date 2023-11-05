import os

import lightgbm as lgb
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from optuna.integration import MLflowCallback
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.constants import RANDOM_STATE

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
        "verbosity": -1,
        "boosting_type": "gbdt",
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


def preprocess():
    df = pd.read_parquet("data/processed/competencia_03.parquet")
    X = df.copy().drop(columns=["clase_ternaria"], axis=1)
    y = df["clase_ternaria"].copy()

    return X, y


def training_loop(X, y):
    mlflow.lightgbm.autolog()

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)
    dtrain = lgb.Dataset(X_train.values, label=y_train)
    dvalid = lgb.Dataset(X_test.values, label=y_test, reference=dtrain)

    sampler = TPESampler(seed=RANDOM_STATE)

    with mlflow.start_run():
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize", sampler=sampler
        )
        study.optimize(
            lambda trial: objective(trial, dtrain, dvalid, X_test.values, y_test.values), n_trials=10, callbacks=[mlflc]
        )

        model = lgb.train(study.best_params, dtrain, valid_sets=[dtrain, dvalid], valid_names=["train", "valid"])

        preds = model.predict(X_test.values)
        preds = np.rint(preds)
        f_score = f1_score(y_test, preds)
        mlflow.log_metric("f-score", f_score)

        mlflow.lightgbm.log_model(model, "model", input_example=X_test.iloc[[0]])
