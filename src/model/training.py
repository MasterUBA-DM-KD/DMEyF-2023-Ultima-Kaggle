import logging
import os
from typing import Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from lightgbm import Booster
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

from src.constants import (
    COLS_TO_DROP,
    MATRIX_GANANCIA,
    METRIC,
    N_TRIALS_OPTIMIZE,
    NFOLD,
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
    ganancia["preds"] = preds
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
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "verbosity": -1,
        "extra_trees": True,
        "force_row_wise": True,
        "first_metric_only": True,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "min_gain_to_split": 0.0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "max_bin": 31,
        "num_boost_round": 9999,
        "seed": RANDOM_STATE,
        "extra_seed": RANDOM_STATE_EXTRA,
        # "num_boost_round": trial.suggest_int("num_boost_round", 50, 9995, 5),
        # "max_depth": trial.suggest_int("max_depth", 2, 256),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.5, log=True),
        # "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        # "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 1024),
        # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        # "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 0.9, step=0.1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0, step=0.1),
    }

    eval_hist = lgb.cv(
        params_space,
        dtrain,
        nfold=NFOLD,
        stratified=True,
        feval=calculate_ganancia,
        seed=RANDOM_STATE,
        callbacks=[optuna.integration.LightGBMPruningCallback(trial, metric=METRIC, valid_name="valid")],
    )

    ganancia = eval_hist[f"valid {METRIC}-mean"][-1]
    ganancia_normalizada = ganancia * NFOLD

    return ganancia_normalizada


def find_best_model(dataset_train: lgb.Dataset) -> dict:
    logger.info("Looking for best model")
    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=PRUNER_WARMUP_STEPS)
    mlflow_callback = MLflowCallback(
        tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        metric_name="ganancia-ternaria",
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
        n_jobs=2,
        callbacks=[mlflow_callback],
        gc_after_trial=True,
    )

    return study.best_params


def training_loop(df_train: pd.DataFrame, fine_tune: bool = True) -> Tuple[Booster, str]:
    mlflow.lightgbm.autolog()
    logger.info("Starting training loop")

    X_train = df_train.drop(columns=COLS_TO_DROP, axis=1).copy()
    y_train_ternaria = df_train["clase_ternaria"].copy()
    y_train_binaria = df_train["clase_binaria"].copy()

    train_weights = y_train_ternaria.to_frame()
    train_weights["weights"] = train_weights["clase_ternaria"].map(WEIGHTS_TRAINING)

    dataset_train = lgb.Dataset(X_train, label=y_train_binaria, weight=train_weights["weights"])

    with mlflow.start_run() as _:
        run_name = mlflow.active_run().info.run_name
        logger.info("MLFlow Run %s - Started", run_name)

        if fine_tune:
            params = find_best_model(dataset_train)

        logger.info("Re-training with best params")
        params["verbosity"] = -1
        params["n_jobs"] = -1

        best_model = lgb.train(
            params,
            dataset_train,
            feval=calculate_ganancia,
        )

        logger.info("Saving best model")
        mlflow.lightgbm.log_model(best_model, "model", input_example=X_train.loc[[0]])

        logger.info("Metrics - Training")
        log_metrics(best_model, X_train, y_train_binaria, "training")

        logger.info("MLFlow Run %s - Finished", run_name)

    mlflow.end_run()

    return best_model, run_name


def log_metrics(model: Booster, X: pd.DataFrame, y: pd.Series, label: str) -> None:
    preds = model.predict(X)
    preds = np.rint(preds)
    f_score = f1_score(y, preds)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)

    mlflow.log_metric(f"{label}_f-score", f_score)
    mlflow.log_metric(f"{label}_accuracy", acc)
    mlflow.log_metric(f"{label}_precision", prec)
    mlflow.log_metric(f"{label}_recall", rec)
    mlflow.log_metric(f"{label}_recall", rec)

    cm = confusion_matrix(y, preds, labels=[0, 1], normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig = disp.plot(cmap="Blues").figure_
    plt.savefig(f"{label}_cm.png")
    mlflow.log_figure(fig, f"{label}_cm.png")
