import sys
import pickle
import pandas as pd
import numpy as np
from utils.model import FTMultilayerPerceptron
from utils.data_processing.normalizers import FTStandardScaler
from utils.data_processing.transform_labels import get_labels, labels_to_numbers
from utils.data_processing.one_hot import one_hot_encoder, one_hot_decoder
from utils.data_processing.selection import train_dev_split
from utils.metrics import accuracy, confusion_matrix, precision_recall_specificity_fscore

cols_to_drop = [0]
model_hidden_dims = [20, 20, 20]
random_state = 0
patience = 10

if __name__ == '__main__':

    try:
        df = pd.read_csv('data.csv', header=None)
        df = df.drop(columns=cols_to_drop)

        X = np.array(df.iloc[:, 1:]).T

        labels = get_labels(df.iloc[:, :1])
        y = labels_to_numbers(df.iloc[:, :1], labels)
        y = one_hot_encoder(y, len(labels))

        X_train, X_dev, y_train, y_dev = train_dev_split(X, y, train_size=0.8, random_state=random_state)

        scaler = FTStandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_dev = scaler.transform(X_dev)

        model_dims = model_hidden_dims
        model_dims.insert(0, int(X_train.shape[0]))
        model_dims.append(y_train.shape[0])

        model = FTMultilayerPerceptron(model_dims,\
            batch_size=30,\
            hidden_activation='tanh',\
            random_state=random_state,\
            verbose=50,\
            max_epoch=100000,\
            early_stopping=True,\
            patience=patience)
        model.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)
        model.plot_learning_curve(costs_dev=True)

        y_pred_train = model.predict(X_train)
        y_pred_dev = model.predict(X_dev)
        y_truth_train = one_hot_decoder(y_train)
        y_truth_dev = one_hot_decoder(y_dev)

        print('\n')
        print('Accuracy for training set = ' + str(accuracy(y_truth_train, y_pred_train)))
        print('Accuracy for      dev set = ' + str(accuracy(y_truth_dev, y_pred_dev)))

        inv_labels = {label: value for value, label in labels.items()}
        inv_labels_predicted = {nb: 'Predicted ' + str(label) for nb, label in inv_labels.items()} 
        inv_labels_true = {nb: 'True ' + str(label) for nb, label in inv_labels.items()} 

        print()
        print('Confusion Matrix for the training set:\n')
        print(confusion_matrix(y_truth_train, y_pred_train).rename(columns=inv_labels_true, index=inv_labels_predicted))
        print()
        print('Confusion matrix for the dev set:\n')
        print(confusion_matrix(y_truth_dev, y_pred_dev).rename(columns=inv_labels_true, index=inv_labels_predicted))

        other_metrics_train = precision_recall_specificity_fscore(y_truth_train, y_pred_train)
        other_metrics_dev = precision_recall_specificity_fscore(y_truth_dev, y_pred_dev)
        print()
        print('Other metrics for training set:\n')
        print(other_metrics_train.rename(columns=inv_labels, index=inv_labels))
        print()
        print('Other metrics for dev set:\n')
        print(other_metrics_dev.rename(columns=inv_labels, index=inv_labels))
        f = open('model_scaler_labels.pkl', 'wb')
        pickle.dump((model, scaler, labels), f)

    except Exception as e:
        print('Something went wrong: ' + str(e))
        sys.exit(1)