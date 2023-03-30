mlflow server

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

mlflow models serve -m "mlflow-artifacts:/0/f6a1fd8c84154dfd907795260dfb156e/artifacts/model" -p 1234