import re
from typing import List

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class NumbersDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["text"] = X["text"].apply(self.__drop_stop_numbers)
        return X

    def __drop_stop_numbers(self, text: str) -> str:
        return re.sub(r"\d", "", text)
