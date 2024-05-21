import pickle

import pandas as pd
from joblib import dump, load
from sklearn import set_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from droppers import drop_new_line, drop_stopwords
from embedings_converters import aveage_embedings, convert_to_columns, glove
from helpers import calculate_f1_score
from text_normalizers import lower_text, min_max, porter_stemmer_mine

random_forest_estimator = Pipeline(
    [
        # ("Average embedings", aveage_embedings.AvereageEmbedings()),
        # ("Embedings to features", convert_to_columns.EmbedingsToFeaturesConverter()),
        ("Random Forest Model", RandomForestClassifier()),
    ]
)
