import logging
import os
from typing import Optional, Tuple

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

from src.constants import (
    COLS_TO_DROP,
    EARLY_STOPPING_ROUNDS,
    MATRIX_GANANCIA,
    N_TRIALS_OPTIMIZE,
    PRUNER_WARMUP_STEPS,
    RANDOM_STATE,
    RANDOM_STATE_EXTRA,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

early_stopper = lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True)


def objective(
    trial: optuna.Trial,
    dtrain: lgb.Dataset,
    dvalid: lgb.Dataset,
    X_valid: pd.DataFrame,
    ternaria_for_ganancia: pd.DataFrame,
) -> float:
    params_space = {
        "metric": "auc",
        "objective": "custom",
        "boosting_type": "gbdt",
        "force_row_wise": True,
        "feature_pre_filter": False,
        "first_metric_only": True,
        "boost_from_average": True,
        "verbosity": -1,
        "seed": RANDOM_STATE,
        "n_jobs": -1,
        "extra_trees": True,
        "extra_seed": RANDOM_STATE_EXTRA,
        "max_depth": trial.suggest_int("max_depth", 2, 256),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1.5, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 0.9, step=0.1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 0.9, step=0.1),
    }

    gbm = lgb.train(
        params_space,
        dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            optuna.integration.LightGBMPruningCallback(trial, "auc", "valid"),
            early_stopper,
        ],
    )

    preds = gbm.predict(X_valid, n_jobs=-1)
    ternaria_for_ganancia["preds"] = np.rint(preds)
    ternaria_for_ganancia["ganancia"] = ternaria_for_ganancia["preds"] * ternaria_for_ganancia["weights"]
    ganancia_total = float(ternaria_for_ganancia["ganancia"].sum())

    return ganancia_total


def find_best_model(
    dataset_train: lgb.Dataset, dataset_valid: lgb.Dataset, X_valid: pd.DataFrame, valid_ternaria: pd.DataFrame
) -> dict:
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
        lambda trial: objective(trial, dataset_train, dataset_valid, X_valid, valid_ternaria),
        n_trials=N_TRIALS_OPTIMIZE,
        n_jobs=2,
        callbacks=[mlflow_callback],
        gc_after_trial=True,
    )

    return study.best_params


def training_loop(
    df_train: pd.DataFrame, df_valid: pd.DataFrame, params: Optional[dict] = None
) -> Tuple[LGBMClassifier, str]:
    mlflow.lightgbm.autolog()
    logger.info("Starting training loop")

    valid_ternaria = df_valid["clase_ternaria"].copy()
    valid_ternaria = valid_ternaria.to_frame()
    valid_ternaria["weights"] = valid_ternaria["clase_ternaria"].map(MATRIX_GANANCIA)

    X_train = df_train.drop(columns=COLS_TO_DROP, axis=1).copy()
    X_valid = df_valid.drop(columns=COLS_TO_DROP, axis=1).copy()

    y_train = df_train["clase_binaria"].copy()
    y_valid = df_valid["clase_binaria"].copy()

    weights = y_train.value_counts(normalize=True).min() / y_train.value_counts(normalize=True)
    train_weights = (
        pd.DataFrame(y_train.rename("old_target"))
        .merge(weights, how="left", left_on="old_target", right_on=weights.index)
        .values
    )

    weights = y_valid.value_counts(normalize=True).min() / y_valid.value_counts(normalize=True)
    valid_weights = (
        pd.DataFrame(y_valid.rename("old_target"))
        .merge(weights, how="left", left_on="old_target", right_on=weights.index)
        .values
    )

    dataset_train = lgb.Dataset(X_train, label=y_train, weight=train_weights[:, 1])
    dataset_valid = lgb.Dataset(X_valid, label=y_valid, reference=dataset_train, weight=valid_weights[:, 1])

    with mlflow.start_run() as _:
        run_name = mlflow.active_run().info.run_name
        logger.info("MLFlow Run %s - Started", run_name)

        if params is None:
            params = find_best_model(dataset_train, dataset_valid, X_valid, valid_ternaria)

        logger.info("Re-training with best params")
        params["verbosity"] = -1
        best_model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE)
        best_model = best_model.fit(
            X_train,
            y_train,
            eval_metric="auc",
            eval_names=["train", "valid"],
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            callbacks=[early_stopper],
        )

        preds = best_model.predict(X_valid, n_jobs=-1)
        preds = np.rint(preds)
        f_score = f1_score(y_valid, preds)
        mlflow.log_metric("f-score", f_score)

        logger.info("Saving best model")
        mlflow.lightgbm.log_model(best_model, "model")

        logger.info("Metrics - Training")
        log_metrics(best_model, X_train, y_train, "training")

        logger.info("Metrics - Validation")
        log_metrics(best_model, X_valid, y_valid, "validation")

        logger.info("MLFlow Run %s - Finished", run_name)

    mlflow.end_run()

    return best_model, run_name


def log_metrics(model: lgb.LGBMClassifier, X: pd.DataFrame, y: pd.Series, label: str) -> None:
    preds = model.predict(X, n_jobs=-1)
    preds = np.rint(preds)
    f_score = f1_score(y, preds)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)

    mlflow.log_metric(f"{label}_f-score", f_score)
    mlflow.log_metric(f"{label}_accuracy", acc)
    mlflow.log_metric(f"{label}_precision", prec)
    mlflow.log_metric(f"{label}_recall", rec)

    cm = confusion_matrix(y, preds, labels=model.classes_, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig = disp.plot(cmap="Blues").figure_
    plt.savefig(f"{label}_cm.png")
    mlflow.log_figure(fig, f"{label}_cm.png")
