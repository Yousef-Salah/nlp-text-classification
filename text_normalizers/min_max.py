from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class MinMaxNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scaler = MinMaxScaler().fit(X)
        dump(scaler, "trained_models/min_max_scaler.joblib")
        return scaler.transform(X)
