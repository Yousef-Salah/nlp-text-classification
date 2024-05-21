import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class AvereageEmbedings(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["text"] = X["text"].map(lambda embedings: np.mean(embedings, axis=0))
        return X
