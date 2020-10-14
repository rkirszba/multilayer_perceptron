import numpy as np

def one_hot_encoder(y, nb_labels):
    return np.eye(nb_labels)[:, y].reshape(-1, y.shape[1])

def one_hot_decoder(y):
    return np.argmax(y, axis=0).reshape(1, -1)



