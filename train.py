import sys
import pickle
import pandas as pd
import numpy as np
from utils.model import FTMultilayerPerceptron
from utils.data_processing.normalizers import FTStandardScaler
from utils.data_processing.transform_labels import get_labels, labels_to_numbers
from utils.data_processing.one_hot import one_hot_encoder, one_hot_decoder
from utils.data_processing.train_dev_split import train_dev_split
from utils.metrics import accuracy, confusion_matrix, precision_recall_specificity_fscore

cols_to_drop = [0,\
4,\
6,\
10,\
11,\
13,\
16,\
17,\
18,\
19,\
20,\
21,\
24,\
26,\
31]
model_hidden_dims = [7, 7]
keep_probs = [None, 0.6, 0.6, None]
random_state = 0

if __name__ == '__main__':

   # try:
        df = pd.read_csv('data.csv', header=None)
        df = df.drop(columns=cols_to_drop)
        X = np.array(df.iloc[:, 1:]).T
        labels = get_labels(df.iloc[:, :1])
        y = labels_to_numbers(df.iloc[:, :1], labels)
        y = one_hot_encoder(y, len(labels))
        X_train, X_dev, y_train, y_dev = train_dev_split(X, y, random_state=random_state)
        scaler = FTStandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_dev = scaler.transform(X_dev)
        model_dims = model_hidden_dims
        model_dims.insert(0, int(X_train.shape[0]))
        model_dims.append(y_train.shape[0])
        model = FTMultilayerPerceptron(model_dims,\
            batch_size=30,\
            random_state=random_state,\
            verbose=50,\
            early_stopping=True,
            dropout_reg=False,
            optimizer='gradient_descent',\
            keep_probs=keep_probs,
            l2_reg=False)
        model.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)
    #    model.plot_learning_curve(costs_dev=True)
        y_pred_train = model.predict(X_train)
        y_pred_dev = model.predict(X_dev)
        y_truth_train = one_hot_decoder(y_train)
        y_truth_dev = one_hot_decoder(y_dev)

        print('Accuracy for training set = ' + str(accuracy(y_truth_train, y_pred_train)))
        print('Accuracy for      dev set = ' + str(accuracy(y_truth_dev, y_pred_dev)))

        print(y_truth_train.shape)
        print(y_pred_train.shape)

        print(y_truth_dev.shape)
        print(y_pred_dev.shape)

        print(pd.DataFrame(confusion_matrix(y_truth_train, y_pred_train)))
        print(confusion_matrix(y_truth_train, y_pred_train))
        print(confusion_matrix(y_truth_dev, y_pred_dev))

        print(precision_recall_specificity_fscore(y_truth_train, y_pred_train))

        f = open('model_scaler_labels.pkl', 'wb')
        pickle.dump((model, scaler, labels), f)

   # except Exception as e:
    #    print('Something went wrong: ' + str(e))
     #   sys.exit(1)