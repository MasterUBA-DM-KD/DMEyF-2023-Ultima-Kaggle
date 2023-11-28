# DM EyF - Última competencia Kaggle

## Setup

Clone the repository:

```bash
git clone git@github.com:UribeAlejandro/DMEyF-2023-Ultima-Kaggle.git
```

Install dependencies:

```bash
pip install -r DMEyF-2023-Ultima-Kaggle/requirements.txt
```

Create a symbolic link to the database folder in GCP (gcsfuse might be active):

```bash
ln -sf ~/buckets/b1/datasets/ DMEyF-2023-Ultima-Kaggle/
```

## MLFlow

Set environment variables:

```bash
export MLFLOW_ARTIFACT_ROOT=gs://mlflow-artifacts-uribe/mlruns
export MLFLOW_TRACKING_URI=sqlite:///database/mlruns.db
```

Start MLFlow server:

```bash
mlflow server --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root $MLFLOW_ARTIFACT_ROOT --host 0.0.0.0 --port 6000
```

Start MLFlow UI:

```bash
mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI --default-artifact-root $MLFLOW_ARTIFACT_ROOT --host 0.0.0.0 --port 5000
```

### Run ETL - Training

The current pipeline requires a `N1` machine in `GCP`, at least 350GB of RAM and 16 vCPUs. Change any constant in `src/constants.py` if needed. The following command will start the pipeline:

```bash
python3 -m src.main
```

### Upload predictions

After training the model, the predictions can be uploaded to `Kaggle` using the following command:

```bash
kaggle competitions submit -c dmeyf-2023-ultima -f buckets/b1/datasets/processed/predictions/RUN_NAME/CUT.csv -m "mlflow: RUN_NAME"
```

Make sure to change `RUN_NAME` and `CUT` to the desired values.
