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

class KFold():

    def __init__(self, X, y, k, random_state=None):
        self.k_ = k
        self.random_state_ = random_state
        self.X_shuffled_, self.y_shuffled_ = self._shuffle(X, y)
        self.curs_ = 0
        self.m_ = X.shape[1]
        self.fold_size_ = X.shape[1] // k


    def __iter__(self):
        return self

    def _shuffle(self, X, y):
        m = X.shape[1]
        if self.random_state_ is not None:
            np.random.seed(self.random_state_)
        permutation = np.random.permutation(m)
        return X[:, permutation].reshape(-1, m),\
            y[:, permutation].reshape(-1, m)

    def __next__(self):
        if self.curs_ == self.k_:
            self.curs_ = 0
            raise StopIteration
        if self.curs_ == self.k_ - 1:
            dev_indices = [i for i in range(self.m_) if i >= self.fold_size_ * self.curs_]
        else:
            dev_indices = [i for i in range(self.m_) if i >= self.fold_size_ * self.curs_ and i < self.fold_size_ * (self.curs_ + 1)]
        train_indices = [i for i in range(self.m_) if i not in dev_indices]
        self.curs_ += 1
        return self.X_shuffled_[:,train_indices], self.X_shuffled_[:,dev_indices],\
            self.y_shuffled_[:,train_indices], self.y_shuffled_[:,dev_indices]


