import numpy as np

def get_labels(df):
    labels = {}
    for i, label in enumerate(df[df.columns[0]].unique()):
        labels[label] = i
    return labels

def labels_to_numbers(df, labels):
    y = np.array(df[df.columns[0]].map(labels)).reshape(1, -1)
    return y