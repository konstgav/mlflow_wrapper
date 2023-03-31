
# Обертка над `mlflow` моделью

## Шаги по тестированию работоспособности модели

1. Установка `mlflow`
```
pip install mlflow==1.25.1
```

2. Запуск локального `mlflow`-сервера.
```
mlflow server
```
Веб-интерфейс доступен по адресу [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. Обучение и логирование модели 
```
python3 wrapper_test.py
```

4. В терминале, где будет запускаться модель, указать путь к `mlflow`-серверу
```
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

5. Запуск модели (serving), путь к модели нужно заменить (2 минуты).
```
mlflow models serve -m "./mlruns/0/7fe77989175146369b941d5feaea1cc1/artifacts/model" -p 1234
```

6. Проверка работоспособности модели
```
curl -X POST -H "Content-Type:application/json" --data '{"dataframe_split": {"columns":["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],"data":[[6.2, 0.66, 0.48, 1.2, 0.029, 29, 75, 0.98, 3.33, 0.39, 12.8]]}}' http://127.0.0.1:1234/invocations
```

7. Появилось новое `conda`-окружение
```
conda env list
```

## Для версии mlflow 2.2.2
Есть проблема с версией `mlflow` 2.2.2 -- Образ с окружением модели не стартует.
В старых версиях `mlflow` создавалось `conda`-окружение, в которое загружались и устанавливались нужные для модели пакеты.
В ноых версиях `mlflow` используется пакет `virtualenv` для создания виртуального окружения, из-за этого возникает ошибка.

Дополнительные шаги
```
curl https://pyenv.run | bash
nano ~/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

pip install virtualenv

sudo apt-get install libbz2-dev
```

## Refs
1. [https://medium.com/@pennyqxr/how-save-and-load-fasttext-model-in-mlflow-format-37e4d6017bf0]()