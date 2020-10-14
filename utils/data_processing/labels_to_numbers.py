import numpy as np

def labels_to_numbers(df):
    labels = {}
    for i, label in enumerate(df[df.columns[0]].unique()):
        labels[label] = i
    y = np.array(df[df.columns[0]].map(labels)).reshape(1, -1)
    return y, labels