from typing import List

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class WordOfKLengthDropper(BaseEstimator, TransformerMixin):
    def __init__(self, k=2) -> None:
        super().__init__()
        self.__k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["text"] = X["text"].apply(self.__drop_words_of_k_length)
        return X

    def __drop_words_of_k_length(self, text: str) -> str:
        words = word_tokenize(text)

        return " ".join([word for word in words if len(word) >= self.__k])
