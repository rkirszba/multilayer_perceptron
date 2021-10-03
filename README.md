# multilayer_perceptron
42 project


## Description
The goal of this project is to create a neural network from scratch. This neural network will then allow to train a model capable of predicting whether a breast tumor is malignant or benign.


## Installing packages
At the root of the project, you can launch one of the following commands:

```bash
pip install -r requirements.txt
```

or

```bash
conda install --file requirements.txt
```

## Features

### Multilayer Perceptron
The FTMultilayerPerceptron class allows to do the following:
- choice between 3 parameters initilizations
- 4 optimizers (gradient descent, momentum, rmsprop, adam)
- choice between 4 hidden layers activation functions (relu, lrelu, tanh, sigmoid)
- drop out and / or l2 regularization
- learning rate decay
- early stopping
- plotting the learning curves
- verbose

### Processing data utils
- one hot encoder for labels
- 2 normalizers (FTStandardScaler, FTMinmaxScaler classes)
- train dev splitter
- KFold class

### Metrics
- cross entropy cost
- accuracy
- confusion matrix
- precision, recall, specificity, fscore

## Use

### Training

```bash
python3 train.py
```

This program does the following:
- reading the data.csv file
- preprocessing it
- normalizing data
- setting the parameters of the neural network
- training the model
- displaying learning curves and metrics
- dumping model and normalizer in pickle file

### Predicting

```bash
python3 predict.py [file].csv
```

This program does the following:
- getting model and normalizer from the pickle file
- reading the data.csv file
- preprocessing it
- normalizing data
- making a prediction on the data
- evaluating it thanks to multiple metrics
