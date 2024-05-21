from typing import List

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class StopWordsDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["text"] = X["text"].apply(self.__drop_stop_words)
        return X

    def __drop_stop_words(self, text: str) -> str:
        words = self.__tokinize(text)
        return " ".join(
            [word for word in words if word not in stopwords.words("english")]
        )

    def __tokinize(self, text) -> List[str]:
        return word_tokenize(text)
