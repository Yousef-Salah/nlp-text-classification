from sklearn.base import BaseEstimator, TransformerMixin


class NewlineDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["text"] = X["text"].apply(lambda text: text.replace("\n", " "))
        return X
