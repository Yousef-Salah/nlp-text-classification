import pickle
import time

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from joblib import dump, load
from sklearn import set_config
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from droppers import drop_new_line, drop_stopwords
from embedings_converters import aveage_embedings, convert_to_columns, glove
from helpers import calculate_f1_score
from pipelines import (
    gradient_boosting_pipeline,
    knn_pipeline,
    naive_bayes_pipeline,
    random_forest_pipeline,
    svm_pipeline,
    text_preprocessing_pipeline,
)
from text_normalizers import lower_text, min_max, porter_stemmer_mine

# TODO: Apply singleton DP for `word tokinzer`
text_processor_pipeline = text_preprocessing_pipeline.text_processor_pipeline

start_time = time.time()

df = pd.read_json("training.json")
X_train = df.drop(["category"], axis=1)
Y_train = df[["category"]]


X_transformed = text_processor_pipeline.fit_transform(X_train)


exit()

estimators = {
    "SVM": svm_pipeline.svm_estimator,
    "random_forest": random_forest_pipeline.random_forest_estimator,
    "naive_beyes": naive_bayes_pipeline.naive_bayes_estimator,
    "KNN_3": KNeighborsClassifier(n_neighbors=3),
    "KNN_5": KNeighborsClassifier(n_neighbors=5),
    "KNN_7": KNeighborsClassifier(n_neighbors=7),
    # "gradient_boosting": gradient_boosting_pipeline.gradien_boosting_estimator,
}

for model_name, estimator in estimators.items():
    print(f"{model_name} start learning")
    model = estimator.fit(X_transformed, Y_train)
    print(f"{model_name} finish learning")
    dump(model, f"trained_models/{model_name}_lema.joblib")
    print(f"{model_name} learned model saved successfully")
    print("\n" * 3)

models = {
    "SVM": "SVM.joblib",
    "random_forest": "random_forest.joblib",
    "naive_beyes": "naive_beyes.joblib",
    "KNN_3": "KNN_3.joblib",
    "KNN_5": "KNN_5.joblib",
    "KNN_7": "KNN_7.joblib",
    # "gradient_boosting": "gradient_boosting.joblib",
}


X_test = pd.read_json("testing.json")
Y_test = X_test[["category"]]
X_transformed = text_processor_pipeline.fit_transform(X_test)
Y_test = Y_test.values.ravel()
print("Transformation phase done.")

trained_scaller = load("trained_models/min_max_scaler.joblib")

for model_name, model_path in models.items():
    model = load(f"trained_models/{model_name}_lema.joblib")

    Y_predicted = model.predict(X_transformed)

    score = f1_score(Y_test, Y_predicted, average="macro")

    print(f"{model} score is = {score}")
