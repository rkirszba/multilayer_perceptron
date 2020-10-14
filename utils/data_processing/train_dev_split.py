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