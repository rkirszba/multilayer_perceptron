import sys
import numpy as np
import pandas as pd
import pickle
from normalizer import Normalizer
from ft_multilayer_perceptron import FTMultilayerPerceptron

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python logreg_predict.py resources/dataset_test.csv')
        sys.exit(1)

    dataset_orig = pd.read_csv(sys.argv[1])
    f = open('classifier_normalizer.pkl', 'rb')
    classifier, normalizer = pickle.load(f)
    dataset = dataset_orig.iloc[:, [7, 8, 10, 11, 12, 13, 14, 15, 17, 18]]
    
    description = dataset.describe()
    cols = list(dataset.columns)
    values = {}
    for col in cols:
        values[col] = description[col]['mean']
    dataset = dataset.fillna(value=values)

    X = np.array(dataset)
    X = normalizer.normalize(X).T
    y_pred = classifier.predict(X)

    label_map = {
        0: 'Ravenclaw',
        1: 'Slytherin',
        2: 'Gryffindor',
        3: 'Hufflepuff'
    }
    df = pd.DataFrame(y_pred.T.reshape(-1,))
    df[0] = df[0].map(label_map)
    df = df.rename(columns={0:'Hogwarts House'})
    df.to_csv(path_or_buf='houses.csv', index_label='Index')

