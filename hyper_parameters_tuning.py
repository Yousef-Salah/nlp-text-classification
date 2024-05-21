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
from sklearn.svm import LinearSVC

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

df = pd.read_json("training-emb.json")
X_train = df.drop(["category"], axis=1)
Y_train = df[["category"]]
Y_train = Y_train.values.ravel()

ros = RandomOverSampler(random_state=42)
# X_train_resampled, Y_train_resampled = ros.fit_resample(X_train, Y_train)


df = pd.read_json("testing-emb.json")
X_test = df.drop(["category"], axis=1)
Y_test = df[["category"]]
Y_test = Y_test.values.ravel()

# Random Forest FINE_TUNE
# {'bootstrap': True,
#  'ccp_alpha': 0.0,
#  'class_weight': None,
#  'criterion': 'gini',
#  'max_depth': None,
#  'max_features': 'sqrt',
#  'max_leaf_nodes': None,
#  'max_samples': None,
#  'min_impurity_decrease': 0.0,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'min_weight_fraction_leaf': 0.0,
#  'n_estimators': 100,
#  'n_jobs': None,
#  'oob_score': False,
#  'random_state': None,
#  'verbose': 0,
#  'warm_start': False}
n_estimators = [100, 150, 200, 250, 300]
bootstrap = [True, False]
max_depths = [100, 200, 300, 400]
min_samples_split = [2, 5, 10, 15]

for n_estimator in n_estimators:
    for is_bootstrap in bootstrap:
        for max_depth in max_depths:
            for min_to_split in min_samples_split:
                try:
                    model = RandomForestClassifier(
                        n_estimators=n_estimator,
                        bootstrap=is_bootstrap,
                        max_depth=max_depth,
                        min_samples_split=min_to_split,
                    )
                    model.fit(X_train, Y_train)
                    predictions = model.predict(X_test)
                    score = f1_score(Y_test, predictions, average="macro")
                    print(f"n_estimator is {n_estimator}")
                    print(f"is_bootstrap is {is_bootstrap}")
                    print(f"max_depth is {max_depth}")
                    print(f"min_to_split is {min_to_split}")
                    print(f"f1_score is {score}")
                    print("\n")
                except ValueError as e:
                    continue


# # l2 is not supported with hinge loss
# # duals has no impact
# penalties = ["l1", "l2"]
# losses = ["hinge", "squared_hinge"]
# max_iters = [1000, 1500, 2000]

# for penalty in penalties:
#     for loss in losses:
#         for max_iter in max_iters:

#             if penalty == "l1" and loss == "hinge":
#                 pass
#             else:
#                 try:
#                     model = LinearSVC(
#                         penalty=penalty,
#                         loss=loss,
#                         max_iter=max_iter,
#                     )
#                     model.fit(X_train_resampled, Y_train_resampled)
#                     predictions = model.predict(X_test)
#                     score = f1_score(Y_test, predictions, average="macro")
#                     print(f"Penalty is {penalty}")
#                     print(f"loss is {loss}")
#                     print(f"max_iter is {max_iter}")
#                     print(f"f1_score is {score}")
#                     print("\n")
#                 except ValueError as e:
#                     print(str(e))
#                     continue
