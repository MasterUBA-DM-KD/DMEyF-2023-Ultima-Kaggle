import logging
import os
from typing import Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.integration import MLflowCallback
from optuna.samplers import TPESampler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.constants import EARLY_STOPPING_ROUNDS, EVALUATOR_CONFIG, N_TRIALS_OPTIMIZE, PRUNER_WARMUP_STEPS, RANDOM_STATE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mlflc = MLflowCallback(
    tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
    metric_name="f1_score",
    create_experiment=False,
    mlflow_kwargs={
        "nested": True,
    },
)

early_stopper = lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True)

storage = optuna.storages.RDBStorage(
    url="sqlite:///database/optuna.db",
    engine_kwargs={"connect_args": {"timeout": 120}},
)


def objective(trial: optuna.Trial, dtrain: lgb.Dataset, dvalid: lgb.Dataset, X_test, y_test):
    param = {
        "metric": "auc",
        "objective": "binary",
        "boosting_type": "gbdt",
        "force_col_wise": True,
        "feature_pre_filter": False,
        "verbosity": 1,
        "seed": RANDOM_STATE,
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("lr", 1e-5, 1.5, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 0.9, step=0.1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 0.9, step=0.1),
    }

    gbm = lgb.train(
        param,
        dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            optuna.integration.LightGBMPruningCallback(trial, "auc", "valid"),
            early_stopper,
        ],
    )

    preds = gbm.predict(X_test)
    preds = np.rint(preds)
    metric = f1_score(y_test, preds)

    return metric


def training_loop(df_train: pd.DataFrame, df_valid: pd.DataFrame) -> Tuple[LGBMClassifier, str]:
    logger.info("Starting training loop")
    mlflow.lightgbm.autolog()

    X_train = df_train.drop(columns=["clase_binaria", "foto_mes", "numero_de_cliente"], axis=1)
    y_train = df_train["clase_binaria"]
    weights = y_train.value_counts(normalize=True).min() / y_train.value_counts(normalize=True)
    train_weights = (
        pd.DataFrame(y_train.rename("old_target"))
        .merge(weights, how="left", left_on="old_target", right_on=weights.index)
        .values
    )

    X_valid = df_valid.drop(columns=["clase_binaria", "foto_mes", "numero_de_cliente"], axis=1)
    y_valid = df_valid["clase_binaria"]

    dataset_train = lgb.Dataset(X_train, label=y_train, weight=train_weights[:, 1])
    dataset_valid = lgb.Dataset(X_valid, label=y_valid, reference=dataset_train)

    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=PRUNER_WARMUP_STEPS)

    with mlflow.start_run() as _:
        run_name = mlflow.active_run().info.run_name
        logger.info("MLFlow Run %s - Started", run_name)
        study = optuna.create_study(storage=storage, pruner=pruner, direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: objective(trial, dataset_train, dataset_valid, X_valid.values, y_valid.values),
            n_trials=N_TRIALS_OPTIMIZE,
            n_jobs=1,
            callbacks=[mlflc],
            gc_after_trial=True,
        )

        logger.info("Best trial - Retrain")
        best_params = study.best_params
        best_params["learning_rate"] = best_params.pop("lr")

        best_model = lgb.LGBMClassifier(**study.best_params, random_state=RANDOM_STATE, n_jobs=-1)
        best_model = best_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[early_stopper])

        preds = best_model.predict(X_valid)
        preds = np.rint(preds)
        f_score = f1_score(y_valid, preds)
        mlflow.log_metric("f-score", f_score)

        eval_data = X_valid.copy()
        eval_data["target"] = y_valid.copy()

        logger.info("Saving model")
        model_info = mlflow.lightgbm.log_model(best_model, "model")
        mlflow.evaluate(
            model_info.model_uri,
            data=eval_data,
            targets="target",
            model_type="classifier",
            evaluators="default",
            evaluator_config=EVALUATOR_CONFIG,
        )

        logger.info("Metrics - Training")
        log_metrics(best_model, X_train, y_train, "training")

        logger.info("Metrics - Validation")
        log_metrics(best_model, X_valid, y_valid, "validation")

        logger.info("MLFlow Run %s - Finished", run_name)

    return best_model, run_name


def log_metrics(model, X, y, label):
    preds = model.predict(X)
    preds = np.rint(preds)
    f_score = f1_score(y, preds)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    ganancia = 273000 * (preds == 1).sum() - 7000 * ((preds == 0).sum() + (preds == 1).sum())
    mlflow.log_metric(f"{label}_f-score", f_score)
    mlflow.log_metric(f"{label}_accuracy", acc)
    mlflow.log_metric(f"{label}_precision", prec)
    mlflow.log_metric(f"{label}_recall", rec)
    mlflow.log_metric(f"{label}_ganancia", ganancia)
    cm = confusion_matrix(y, preds, labels=model.classes_, normalize="all")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig = disp.plot().figure_
    plt.savefig(f"{label}_cm.png")
    mlflow.log_figure(fig, f"{label}_cm.png")
