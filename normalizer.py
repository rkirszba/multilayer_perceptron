import numpy as np

def _means(X):
    return (X.sum(axis=0) / X.shape[0]).reshape(1, -1)

def _standard_deviations(X):
   return np.sqrt((1 / X.shape[0]) * np.sum(np.square(X - _means(X)), axis=0).reshape(1, -1))

def _x_min(X):
    return np.min(X, axis=0).reshape(1, -1)

def _x_max(X):
    return np.max(X, axis=0).reshape(1, -1)

class Normalizer():

    def __init__(self, X, method='z_score'):
        self.means_ = _means(X)
        self.method_ = method
        self.sigmas_ = _standard_deviations(X)
        self.sigmas_ = np.where(self.sigmas_ == 0, 1, self.sigmas_)
        self.x_min_ = _x_min(X)
        self.x_max_ = _x_max(X)
        self.range_ = self.x_max_ - self.x_min_
        self.range_ = np.where(self.range_ == 0, 1, self.range_)

    
    def normalize(self, X):
        if self.method_ == 'z_score':
            return (X - self.means_) / self.sigmas_
        if self.method_ == 'min_max':
            return (X - self.x_min_) / self.range_
        