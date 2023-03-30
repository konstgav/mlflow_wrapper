https://medium.com/@pennyqxr/how-save-and-load-fasttext-model-in-mlflow-format-37e4d6017bf0

pip install mlflow==1.25.1

mlflow server

python3 wrapper_test.py 

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

curl https://pyenv.run | bash

nano ~/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

pip install virtualenv

sudo apt-get install libbz2-dev

mlflow models serve -m "./mlruns/0/551f03bf34534a5296c5d663c3c757e4/artifacts/model" -p 1234

curl -X POST -H "Content-Type:application/json" --data '{"dataframe_split": {"columns":["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],"data":[[6.2, 0.66, 0.48, 1.2, 0.029, 29, 75, 0.98, 3.33, 0.39, 12.8]]}}' http://127.0.0.1:1234/invocations