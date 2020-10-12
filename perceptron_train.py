import sys
import numpy as np
import pandas as pd
import pickle
from normalizer import Normalizer
from ft_multilayer_perceptron import FTMultilayerPerceptron

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python test_perceptron.py resources/dataset_train.csv')
        sys.exit(1)

    #try:
    dataset_orig = pd.read_csv(sys.argv[1])
    
    descriptions = {}
    for house in pd.unique(dataset_orig['Hogwarts House']):
        descriptions[house] = dataset_orig[dataset_orig['Hogwarts House'] == house].describe()

    dataset = dataset_orig.iloc[:, [1, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18]]
    cols = list(dataset.columns)

    for index, row in dataset.iterrows():
        for col in cols:
            if pd.isnull(dataset[col][index]):
                dataset.at[index, col] = descriptions[dataset['Hogwarts House'][index]][col]['mean']

    X = np.array(dataset.iloc[:, 1:])
    normalizer = Normalizer(X)
    X = normalizer.normalize(X).T
    label_map = {
        'Ravenclaw': 0,
        'Slytherin': 1,
        'Gryffindor': 2,
        'Hufflepuff': 3
    }
    y = dataset['Hogwarts House'].map(label_map)
    y = np.array(y).reshape(1, -1)
    y = np.eye(4)[:, y].reshape(-1, y.shape[1])
    classifier = FTMultilayerPerceptron([X.shape[0], 5, 7, 4], verbose=40, max_epoch=1000, optimizer='momentum', random_state=42).fit(X, y)
    '''
    print('\n')
    classifier = FTMultilayerPerceptron([X.shape[0], 5, 7, 4], verbose=40, max_epoch=1000, optimizer='gradient_descent', random_state=42).fit(X, y)
    print('\n')
    classifier = FTMultilayerPerceptron([X.shape[0], 5, 7, 4], verbose=40, max_epoch=1000, optimizer='momentum', random_state=42).fit(X, y)
    print('\n')
    classifier = FTMultilayerPerceptron([X.shape[0], 5, 7, 4], verbose=40, max_epoch=1000, optimizer='rmsprop', random_state=42).fit(X, y)
    print('\n')
    classifier = FTMultilayerPerceptron([X.shape[0], 5, 7, 4], verbose=40, max_epoch=1000, optimizer='adam', random_state=42).fit(X, y)
    '''
  #  classifier.plot_learning_curve()
    classifier.predict
    
    f = open('classifier_normalizer.pkl', 'wb')
    pickle.dump((classifier, normalizer), f)

