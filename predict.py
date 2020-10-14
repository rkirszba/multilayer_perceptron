import sys
import pickle
import pandas
import numpy
from utils.data_processing.transform_labels import labels_to_numbers
from utils.data_processing.one_hot import one_hot_encoder, one_hot_decoder
from utils.metrics import *

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

if __name__ == '__main__':

    try:
        if len(sys.argv) != 2:
            print("Usage: python3 predict.py [file].csv")
            sys.exit(1)
        df = pd.read_csv(sys.argv[1])

        f = open('model_scaler_labels.pkl', rb)
        model, scaler, labels = pickle.load(f)
        
        df = df.drop(columns=cols_to_drop)
        X_test = np.array(df.iloc[:, 1:]).T
        y_test_orig = labels_to_numbers(df.iloc[:, :1], labels)
        y_test_ = one_hot_encoder(y_test_orig, len(labels))
        X_test = scaler.transform(X_test)

        y_hat = model.predict_probas(X_test)
        y_pred = model.predict(X_test)

        cost1 = cross_entropy_cost_alter(y_test, y_hat)
        cost2 = cross_entropy_cost(y_test, y_hat)

        accuracy = accuracy(y_test_orig, y_pred)

        confusion_matrix = confusion_matrix(y_test_orig, y_pred)

        other_metrics = precision_recall_specificity_fscore(y_test_orig, y_pred)

        

        
