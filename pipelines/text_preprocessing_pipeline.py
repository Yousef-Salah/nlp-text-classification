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

from droppers import (
    drop_new_line,
    drop_non_latters,
    drop_numbers,
    drop_stopwords,
    drop_symbol,
    drop_words,
)
from embedings_converters import aveage_embedings, convert_to_columns, glove
from helpers import calculate_f1_score
from text_normalizers import lematizer, lower_text, min_max, porter_stemmer_mine

text_processor_pipeline = Pipeline(
    [
        ("lower text", lower_text.LowerTextConverter()),
        ("New line Dropper", drop_new_line.NewlineDropper()),
        ("drop some symbols", drop_symbol.SymbolsDropper()),
        ("Stopwords Dropper", drop_stopwords.StopWordsDropper()),
        # ("drop non latters characters", drop_non_latters.NonLattersDropper()),
        ("Dro numbers", drop_numbers.NumbersDropper()),
        ("Drop words of length 2 or less", drop_words.WordOfKLengthDropper()),
        # ("Porter Stemmer", porter_stemmer_mine.PorterStemmerMine()),
        ("lematizer", lematizer.Lematizer()),
        ("To Glove Embedings", glove.GloveEmbedingsConverter()),
    ]
)
