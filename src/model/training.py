# Standard Library Imports
import os

# Third Party Imports
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# docformatter Package Imports
from src.constants import RANDOM_STATE

matplotlib.use("TkAgg")
plt.switch_backend("Agg")

mlflc = MLflowCallback(
    tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
    metric_name="accuracy",
    create_experiment=False,
    mlflow_kwargs={
        "nested": True,
    },
)


def training_loop(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)

    mlflow.lightgbm.autolog()
    with mlflow.start_run() as _:

        @mlflc.track_in_mlflow()
        def objective(trial: optuna.Trial):
            """Objective function to be maximized."""
            param = {
                "objective": "binary",
                "metric": "binary_error",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "seed": RANDOM_STATE,
                "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
                "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
                "num_leaves": trial.suggest_int("num_leaves", 2, 512),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 0, 15),
                "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
            }
            gbm = lgb.LGBMClassifier(**param)
            gbm.fit(X_train, y_train)

            preds = gbm.predict(X_train)
            metric_opt = f1_score(y_train, preds)
            mlflow.log_metric("f1_score_fine_tune_train", metric_opt, step=trial.number)

            preds = gbm.predict(X_test)
            metric_opt = f1_score(y_test, preds)
            mlflow.log_metric("f1_score_fine_tune_test", metric_opt, step=trial.number)

            return metric_opt

        sampler = TPESampler(seed=RANDOM_STATE)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=50, n_jobs=-1)

        # input_example = X_test.iloc[[0]]
        # eval_data = X_test.copy()
        # eval_data["target"] = y_test.copy()
        # evaluator_config = {
        #     "explainability_algorithm": "permutation",
        #     "metric_prefix": "evaluation_",
        #     "pos_label": 1
        # }

        model = lgb.LGBMClassifier(**study.best_params, seed=RANDOM_STATE)
        model.fit(X_train, y_train)
        # preds = model.predict(X_test)
        # metric_opt = f1_score(y_test, preds)

        # model_info = mlflow.lightgbm.log_model(model, "model", input_example=input_example)
        # mlflow.evaluate(
        #     model_info.model_uri,
        #     data=eval_data,
        #     targets="target",
        #     model_type="classifier",
        #     evaluators="default",
        #     evaluator_config=evaluator_config,
        # )
