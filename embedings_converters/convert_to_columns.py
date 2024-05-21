import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class EmbedingsToFeaturesConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(X["text"].apply(eval).head())
        return X
