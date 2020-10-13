import sys
import pickle
import pandas as pd
import numpy as np
from utils.ft_multilayer_perceptron import FTMultilayerPerceptron
from utils.preprocessing.normalizers import FTStandardScaler
from utils.preprocessing.labels_one_hot import one_hot_encode

cols_to_drop = [0]
labels = {
    'B': 0,
    'M': 1
}

if __name__ == '__main__':

    try:
        df = pd.read_csv('data.csv', header=None)
        df = df.drop(columns=cols_to_drop)
        X = np.array(df.iloc[:, 1:]).T
        y = one_hot_encode(np.array(df.iloc[:, :1]), labels)
        
        





    except Exception as e:
        print('Something went wrong: ' + str(e))