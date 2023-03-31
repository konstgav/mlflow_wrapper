import fasttext
import mlflow
from .fast_text_wrapper import FastTextWrapper

def save_model(
    df: pd.DataFrame,
    model_name: str = "model_name",
    corpus_filename: str = "prepared_corpus",
    prepared_content: str = "prepared_text",
):
    """
    Runs the Skipgram model using the corpus created and saved by the 'prepare_corpus' function, saves the model and
    returns the trained model. The output of this function is a trained model in binary file format stored as an
    artifact in MLflow tracking server.

    Args:
        df (pd.DataFrame): Dataframe with text content
        model_name (str): Model will be saved with this name as a .bin file
        corpus_filename (str, optional): Name of the .txt file of the corpus. Defaults to 'prepared_corpus'.
        prepared_content (str, optional): Column name for prepared text. Defaults to 'prepared_text'.

    Returns:
        trained_model: Returns the trained model
    """
    prepare_corpus(df, corpus_filename, prepared_content)
    trained_model = fasttext.train_unsupervised(
        corpus_filename + ".txt", model="skipgram", dim=50, ws=10, epoch=400
    )

    fasttext_model_path = model_name + ".bin"
    trained_model.save_model(fasttext_model_path)

    artifacts = {"fasttext_model_path": fasttext_model_path}
    mlflow_pyfunc_model_path = model_name

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(
            artifact_path=mlflow_pyfunc_model_path,
            python_model=FastTextWrapper(),
            code_path=["./your_code_path"],
            artifacts=artifacts,
        )

    return trained_model