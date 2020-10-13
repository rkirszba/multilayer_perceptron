import numpy as np

def one_hot_encode(y, labels):
    return np.eye(len(labels))[:, y].reshape(-1, y.shape[1])

