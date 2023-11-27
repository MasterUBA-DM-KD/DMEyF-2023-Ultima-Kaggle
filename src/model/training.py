import json
import logging
import os
from typing import Tuple

import lightgbm
import mlflow
import numpy as np
import optuna
import optuna.integration.lightgbm as lgb
import pandas as pd
from lightgbm import Booster
from optuna.integration.mlflow import MLflowCallback
from optuna.samplers import TPESampler

from src.constants import (
    COLS_TO_DROP,
    COST_ENVIO,
    METRIC,
    N_TRIALS_OPTIMIZE,
    PRUNER_WARMUP_STEPS,
    RANDOM_STATE,
    RANDOM_STATE_EXTRA,
    WEIGHTS_TRAINING,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calculate_ganancia(preds: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    metric_name = METRIC
    is_higher_better = True

    label = data.get_label()
    weights = data.get_weight()

    ganancia = pd.DataFrame({"preds": preds, "label": label, "weights": weights})
    ganancia["costo"] = ganancia["weights"].map(COST_ENVIO)
    ganancia["ganancia"] = ganancia["preds"] * ganancia["costo"]
    ganancia_total = float(ganancia["ganancia"].sum())

    return metric_name, ganancia_total, is_higher_better


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


def objective(
    trial: optuna.Trial,
    dtrain: lgb.Dataset,
) -> float:
    params_space = {
        "metric": "custom",
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "verbosity": -1,
        "force_row_wise": True,
        "zero_as_missing": True,
        "first_metric_only": True,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "extra_trees": True,
        "seed": RANDOM_STATE,
        "extra_seed": RANDOM_STATE_EXTRA,
        "save_binary": True,
        "max_bin": 15,
        "neg_bagging_fraction": 0.275,
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 256),
        "num_leaves": trial.suggest_int("num_leaves", 8, 70),
    }

    gbm = lightgbm.train(
        params_space,
        dtrain,
        num_boost_round=trial.suggest_int("num_boost_round", 100, 500, step=100),
        feval=calculate_ganancia,
    )

    mlflow.lightgbm.log_model(gbm, f"model_{str(trial.number)}")

    trial.set_user_attr(key="best_booster", value=gbm)

    preds = gbm.predict(dtrain.get_data(), n_jobs=-1)
    _, ganancia, _ = calculate_ganancia(preds, dtrain)

    return ganancia


def find_best_model(dataset_train: lgb.Dataset) -> Booster:
    logger.info("Looking for best model")
    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=PRUNER_WARMUP_STEPS)
    mlflow_callback = MLflowCallback(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        metric_name=METRIC,
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
        callbacks=[mlflow_callback, callback],
        gc_after_trial=True,
    )

    best_model = study.user_attrs["best_booster"]

    return best_model


def training_loop(df_train: pd.DataFrame) -> Tuple[Booster, str]:
    logger.info("Starting training loop")
    mlflow.set_experiment("Fine-Tune")
    mlflow.lightgbm.autolog(log_input_examples=True, log_datasets=False)
    with mlflow.start_run(nested=True):
        run_name = mlflow.active_run().info.run_name
        X_train = df_train.drop(columns=COLS_TO_DROP, axis=1).copy()
        y_train_ternaria = df_train["clase_ternaria"].copy()
        y_train_binaria = df_train["clase_binaria"].copy()

        train_weights = y_train_ternaria.to_frame()
        train_weights["weights"] = train_weights["clase_ternaria"].map(WEIGHTS_TRAINING)

        dataset_train = lgb.Dataset(
            X_train, label=y_train_binaria, weight=train_weights["weights"], free_raw_data=False
        )

        best_model = find_best_model(dataset_train)

        params = best_model.params

        # log model parameters as artifact
        with open("best_parameters.json", "w") as outfile:
            json.dump(params, outfile)

        mlflow.log_artifact("best_parameters.json", "best_model")
        mlflow.lightgbm.log_model(best_model, "best_model", input_example=X_train.loc[[0]])

    mlflow.end_run()

    return best_model, run_name
