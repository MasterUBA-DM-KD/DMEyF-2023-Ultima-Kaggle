# import os
# import mlflow
# os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///database/mlruns.db"
# os.environ["MLFLOW_ARTIFACT_ROOT"] = "mlruns"
#
#
# def test_inf():
#     logged_model = 'runs:/646bddf9e8234e25b3c57a7a6c3321d5/model'
#     loaded_model = mlflow.lightgbm.load_model(logged_model)
#
#     print(loaded_model.)
#
#     assert 1 == 2
