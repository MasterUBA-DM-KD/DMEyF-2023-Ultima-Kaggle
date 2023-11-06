# DM EyF - Última competencia Kaggle

## MLFlow

Set environment variables:

```bash
export MLFLOW_ARTIFACT_ROOT=gs://mlflow-artifacts-uribe/mlruns
export MLFLOW_TRACKING_URI=sqlitdatabase/mlruns.db
```

Start MLFlow server:

```bash
mlflow server --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root $MLFLOW_ARTIFACT_ROOT --host 0.0.0.0 --port 6000
```

Start MLFlow UI:

```bash
mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root $MLFLOW_ARTIFACT_ROOT --host 0.0.0.0 --port 5000
```
