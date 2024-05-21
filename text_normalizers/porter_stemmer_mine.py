from typing import List

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from pandas import Series
from sklearn.base import BaseEstimator, TransformerMixin


class PorterStemmerMine(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["text"] = X.apply(self.__stem, axis=1)
        return X

    def __stem(self, row: Series) -> str:
        ps = PorterStemmer()
        words = self.__tokinize(row["text"])
        words = [ps.stem(word) for word in words]

        return " ".join(words)

    def __tokinize(self, text) -> List[str]:
        return word_tokenize(text)
