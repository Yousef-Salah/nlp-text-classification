import re
from typing import List

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class SymbolsDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["text"] = X["text"].apply(lambda text: re.sub(r"[\/\-\\._]", " ", text))
        return X
