import numpy as np

def train_dev_split(X, y, train_size=0.8, random_state=None):
    m = X.shape[1]
    if random_state is not None:
        np.random.seed(random_state)
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation].reshape(-1, m)
    y_shuffled = y[:, permutation].reshape(-1, m)
    train_nb = int(train_size * m)
    X_train = X_shuffled[:, :train_nb]
    X_dev = X_shuffled[:, train_nb:]
    y_train = y_shuffled[:, :train_nb]
    y_dev = y_shuffled[:, train_nb:]
    return X_train, X_dev, y_train, y_dev

class Kfold():

    def __init__(self, k, random_state=None):
        self.k_ = k
        self.random_state_ = random_state

    def _kfold(self, X, y):
        m = X.shape[1]
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation].reshape(-1, m)
        y_shuffled = y[:, permutation].reshape(-1, m)
        fold_size = m // self.k_
        X_folds = []
        y_folds = []
        for i in range(self.k_):
            if i == self.k_ - 1:
                X_folds.append(X_shuffled[:, i * fold_size :])
                y_folds.append(X_shuffled[:, i * fold_size :])
            else:
                X_folds.append(X_shuffled[:, i * fold_size : (i + 1) * fold_size])
                y_folds.append(X_shuffled[:, i * fold_size : (i + 1) * fold_size])
        return X_folds, y_folds


    def get_splits(self, X, y):
        if self.random_state_ is not None:
            np.random.seed(self.random_state_)
        X_folds, y_folds = self._kfold
        splits = []


