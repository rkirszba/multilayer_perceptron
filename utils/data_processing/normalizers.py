import numpy as np



class FTStandardScaler():

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def _fit(self, X):
        m = X.shape[1]
        self.mean_ = (X.sum(axis=1) / m).reshape(-1, 1)
        self.std_ = np.sqrt((1 / m) * np.sum(np.square(X - self.mean_), axis=1).reshape(-1, 1))
        self.std_ = np.where(self.std_ == 0, 1, self.std_)

    def fit(self, X):
        self._fit(X)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self._fit(X)
        return (X - self.mean_) / self.std_


class FTMinMaxScaler():

    def __init__(self):
        self.x_min_ = None
        self.x_max_ = None

    def _fit(self, X):
        self.x_min_ = X.min(axis=1).reshape(-1, 1)
        self.x_max_ = X.max(axis=1).reshape(-1, 1)
        self.range_ = self.x_max_ - self.x_min_
        self.range_ = np.where(self.range_ == 0, 1, self.range_)

    def transform(self, X):
        return (X - self.x_min_) / self.range_

    def fit_transform(self, X):
        self._fit(X)
        return (X - self.x_min_) / self.range_