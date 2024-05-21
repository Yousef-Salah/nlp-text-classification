import pandas as pd
from joblib import load
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


# TODO: write a function comment block
def calculate_f1_score(
    file_path: str,
    model_path: str,
    pipeline_object: Pipeline,
    target_variable: str = "category",
) -> float:
    """"""
    df = pd.read_json(file_path)
    X_test = df.drop([target_variable], axis=1)
    Y_actual = df[[target_variable]].values.ravel()

    Y_transformed = pipeline_object.fit_transform(X_test)
    model = load(model_path)

    Y_predicted = model.predict(Y_transformed)
    score = f1_score(Y_actual, Y_predicted, average="macro")
    return score
