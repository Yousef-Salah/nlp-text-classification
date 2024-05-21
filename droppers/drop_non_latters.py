import re
from typing import List

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class NonLattersDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["text"] = X["text"].apply(self.__drop_stop_words)
        return X

    def __drop_stop_words(self, text: str) -> str:
        return re.sub(r"[^a-zA-Z]", "", text)
