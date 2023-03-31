import mlflow
import mlflow.pyfunc
import pickle
import logging
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_model(alpha=0.5, l1_ratio=0.5):
    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)
    
    predicted_qualities = lr.predict(test_x)
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    
    return lr, rmse, mae, r2

def trian_and_save_model(model_name, alpha=0.5, l1_ratio=0.5):
    """Train and serialize the model of wine preferences by data mining from physicochemical properties.
    
    Returns:
        trained_model: Returns the trained model
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    with mlflow.start_run() as run:
        sklearn_model, rmse, mae, r2 = train_model(alpha, l1_ratio)

        filename = model_name + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(sklearn_model, f)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        print(rmse, mae, r2)
        artifacts = {"sklearn_model_path": filename}
        mlflow_pyfunc_model_path = model_name

        mlflow.pyfunc.log_model(
            artifact_path=mlflow_pyfunc_model_path,
            python_model=ElasticNetWineModel(),
            #code_path=["./your_code_path"],
            artifacts=artifacts,
        )

    return sklearn_model

def load_model(model_name: str = "model_name", stage: str = "Production"):
    """
    Loads the trained ElasticNet (sklearn) model from the model registry and returns
    it as mlflow.pyfunc.PyFuncModel to be used to do the predictions.
    """
    loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")
    return loaded_model

class ElasticNetWineModel(mlflow.pyfunc.PythonModel):
    """
    Modeling wine preferences by data mining from physicochemical properties.
    """

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.

        Args:
            context: MLflow context where the model artifact is stored.
        """
        filename = context.artifacts["sklearn_model_path"]
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
        
    def predict(self, context, model_input):
        """This is an abstract function. We customized it into a method to fetch the FastText model.

        Args:
            context ([type]): MLflow context where the model artifact is stored.
            model_input ([type]): the input data to fit into the model.

        Returns:
            [type]: the loaded model artifact.
        """
        inference_data = self.model.predict(model_input)
        return inference_data

def test_prediction(model):
    data = [[6.2, 0.66, 0.48, 1.2, 0.029, 29, 75, 0.98, 3.33, 0.39, 12.8]]
    columns=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
    df = pd.DataFrame(data, columns=columns)
    result = model.predict(df)
    return result


if __name__ == '__main__':
    trian_and_save_model(model_name='ElasticNetWineModel')
    loaded_model = load_model()
    result = test_prediction(loaded_model)
    print(result)