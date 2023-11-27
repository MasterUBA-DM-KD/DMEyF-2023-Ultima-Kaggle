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
    BASE_PATH_PREDICTIONS,
    COLS_TO_DROP,
    COST_ENVIO,
    FINE_TUNE,
    METRIC,
    N_TRIALS_OPTIMIZE,
    PARAMS,
    PRUNER_WARMUP_STEPS,
    RANDOM_STATE,
    RANDOM_STATE_EXTRA,
    SEEDS,
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
        "max_bin": 30,
        "num_leaves": 67,
        "neg_bagging_fraction": 0.275,
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 256),
    }

    gbm = lightgbm.train(
        params_space,
        dtrain,
        num_boost_round=trial.suggest_int("num_boost_round", 450, 700, step=50),
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


def training_loop(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[Booster, str]:
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

        if FINE_TUNE:
            best_model = find_best_model(dataset_train)
            params = best_model.params
            # log model parameters as artifact
            with open("best_parameters.json", "w") as outfile:
                json.dump(params, outfile)

            mlflow.log_artifact("best_parameters.json", "best_model")
            mlflow.lightgbm.log_model(best_model, "best_model", input_example=X_train.loc[[0]])
        else:
            best_model = semillero(dataset_train, df_test, run_name)
    mlflow.end_run()

    return best_model, run_name


def semillero(dataset_train: lgb.Dataset, df_test: pd.DataFrame, run_name: str) -> Booster:
    logger.info("Starting predictions per seed")

    base_path = os.path.join(BASE_PATH_PREDICTIONS, run_name)
    os.makedirs(base_path, exist_ok=True)

    model = None
    X_test = df_test.drop(columns=COLS_TO_DROP, axis=1).copy()
    final_preds = df_test["numero_de_cliente"].to_frame()

    for seed in SEEDS:
        logger.info("Training with seed %s", seed)
        PARAMS["seed"] = seed
        if model is not None:
            model = lightgbm.train(
                train_set=dataset_train,
                params=PARAMS,
                init_model=model,
                feval=calculate_ganancia,
            )
        else:
            model = lightgbm.train(
                train_set=dataset_train,
                params=PARAMS,
                feval=calculate_ganancia,
            )

        logger.info("Prediction with seed %s", seed)
        preds = model.predict(X_test)
        final_preds[f"seed_{seed}"] = preds

    final_preds["Predicted"] = final_preds.iloc[:, 1:].mean(axis=1)

    logger.info("Aggregating predictions - all seeds")
    final_preds = final_preds.sort_values(by="Predicted", ascending=False)
    final_preds = final_preds.reset_index(drop=True)
    final_preds.to_csv(os.path.join(base_path, "predictions.csv"), index=False)

    logger.info("Saving aggregated predictions")
    for cut in range(5000, 20000, 500):
        final_preds_cut = final_preds.copy()
        final_preds_cut = final_preds_cut[["numero_de_cliente", "Predicted"]]
        final_preds_cut.loc[0:cut, "Predicted"] = True
        final_preds_cut.loc[cut:, "Predicted"] = False
        out_path = os.path.join(base_path, f"{cut}.csv")
        final_preds_cut.to_csv(out_path, index=False)

    logger.info("Saved predictions to %s", base_path)
    logger.info("Finished predictions per seed")

    return model
