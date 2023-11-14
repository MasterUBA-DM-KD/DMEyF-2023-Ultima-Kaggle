import logging
import os
from typing import Tuple

import lightgbm as lgb
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from optuna.integration import MLflowCallback
from optuna.samplers import TPESampler

from src.constants import (
    COLS_TO_DROP,
    MATRIX_GANANCIA,
    N_TRIALS_OPTIMIZE,
    PRUNER_WARMUP_STEPS,
    RANDOM_STATE,
    RANDOM_STATE_EXTRA,
    WEIGHTS_TRAINING,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calculate_ganancia(preds: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    metric_name = "ganancia"
    is_higher_better = True

    label = data.get_label()
    weights = data.get_weight()

    ganancia = pd.DataFrame({"preds": preds, "label": label, "weights": weights})
    ganancia["preds"] = np.rint(preds)
    ganancia["costo"] = ganancia["weights"].map(MATRIX_GANANCIA)
    ganancia["ganancia"] = ganancia["preds"] * ganancia["costo"]
    ganancia_total = float(ganancia["ganancia"].sum())

    return metric_name, ganancia_total, is_higher_better


def objective(
    trial: optuna.Trial,
    dtrain: lgb.Dataset,
) -> float:
    params_space = {
        "objective": "binary",
        "metric": "custom",
        "boosting_type": "dart",
        "n_jobs": -1,
        "verbosity": -1,
        "extra_trees": True,
        "force_row_wise": True,
        "first_metric_only": True,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "seed": RANDOM_STATE,
        "extra_seed": RANDOM_STATE_EXTRA,
        "num_boost_round": trial.suggest_int("num_boost_round", 50, 500, 50),
        "max_depth": trial.suggest_int("max_depth", 2, 256),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 1.5, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 1024),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 8000, step=100),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 0.9, step=0.1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0, step=0.1),
    }

    gbm = lgb.train(
        params_space,
        dtrain,
        feval=calculate_ganancia,
    )

    preds = gbm.predict(dtrain.get_data(), n_jobs=-1)
    _, ganancia, _ = calculate_ganancia(preds, dtrain)

    mlflow.log_metric("ganancia", ganancia)

    return ganancia


def find_best_model(dataset_train: lgb.Dataset) -> dict:
    logger.info("Looking for best model")
    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=PRUNER_WARMUP_STEPS)
    mlflow_callback = MLflowCallback(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        metric_name="ganancia",
        create_experiment=False,
        mlflow_kwargs={
            "nested": True,
        },
    )

    study = optuna.create_study(
        pruner=pruner,
        direction="maximize",
        sampler=sampler,
        study_name="Fine-Tune",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, dataset_train),
        n_trials=N_TRIALS_OPTIMIZE,
        n_jobs=1,
        callbacks=[mlflow_callback],
        gc_after_trial=True,
    )

    return study.best_params


def training_loop(df_train: pd.DataFrame) -> None:
    mlflow.lightgbm.autolog()
    logger.info("Starting training loop")

    X_train = df_train.drop(columns=COLS_TO_DROP, axis=1).copy()

    y_train_ternaria = df_train["clase_ternaria"].copy()
    y_train_binaria = df_train["clase_binaria"].copy()

    train_weights = y_train_ternaria.to_frame()
    train_weights["weights"] = train_weights["clase_ternaria"].map(WEIGHTS_TRAINING)

    dataset_train = lgb.Dataset(X_train, label=y_train_binaria, weight=train_weights["weights"], free_raw_data=False)

    with mlflow.start_run() as _:
        run_name = mlflow.active_run().info.run_name
        logger.info("MLFlow Run %s - Started", run_name)
        find_best_model(dataset_train)
