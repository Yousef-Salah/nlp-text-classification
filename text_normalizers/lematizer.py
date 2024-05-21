from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class Lematizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.__lematizer = WordNetLemmatizer()
        return self

    def transform(self, X):
        X["text"] = X["text"].apply(self.__lematize)
        return X

    def __lematize(self, text: str) -> str:
        words = word_tokenize(text)
        words = [self.__lematizer.lemmatize(word) for word in words]
        return " ".join(words)
