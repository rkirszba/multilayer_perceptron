import numpy as np
import pandas as pd



def cross_entropy_cost(y, y_hat, epsilon=1e-8):
    m = y.shape[1]
    return np.squeeze((-1 / m) * np.sum(y * np.log(y_hat + epsilon)))

def cross_entropy_cost_alter(y, y_hat, epsilon=1e-8):
    m = y.shape[1]
    return np.squeeze((-1 / m) * np.sum(y[0] * np.log(y_hat[0] + epsilon)\
        + (1 - y[0]) * np.log(1 - y_hat[0] + epsilon)))

def accuracy(y_true, y_pred):
    correct = ((y_true == y_pred) == True).sum()
    m = y_true.shape[1]
    return correct / m

def confusion_matrix(y_true, y_pred):
    labels = np.unique(np.concatenate((y_true, y_pred), axis=1))
    matrix = {}
    for label1 in labels:
        dic = {}
        for label2 in labels:
            count = (((y_pred == label1) & (y_true == label2)) == True).sum()
            dic[label2] = count
        matrix[label1] = dic
    return pd.DataFrame(matrix).T

def precision_recall_specificity_fscore(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    metrics = {}
    for label in matrix.columns:
        dic = {}
        tp = int(matrix.loc[label, label])
        fp = int(matrix.loc[label, matrix.columns != label].sum())
        tn = int(matrix.drop(label, axis=0).loc[:, matrix.columns != label].sum())
        fn = int(matrix.drop(label, axis=0).loc[:, matrix.columns == label].sum())
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        dic['precision'] = precision
        dic['recall'] = recall
        dic['specificity'] = tn / (tn + fp) if tn + fp != 0 else 0
        dic['fscore'] = 2 * (precision * recall / (precision + recall)) if precision + recall != 0 else 0
        metrics[label] = dic
    return pd.DataFrame(metrics).T
    



            
            
            



