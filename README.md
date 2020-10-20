# multilayer_perceptron

# ft_linear_regression
42 project dslr

## Description
The goal of this project is to create a neural network from scratch. This neural network will then allow to train a model capable of predicting whether a breast tumor is malignant or benign.

## Installing Python 3

This process allows to install python on a mac machine on which you are not root. It is taken from 42AI (https://github.com/42-AI) process.

1. Copy paste the following code into your shell rc file (for, example: `~/.zshrc`).

```bash
function set_conda {
    HOME=$(echo ~)
    INSTALL_PATH="/INSTALL/PATH"
    MINICONDA_PATH=$INSTALL_PATH"/miniconda3/bin"
    PYTHON_PATH=$(which python)
    SCRIPT="Miniconda3-latest-MacOSX-x86_64.sh"
    DL_LINK="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"

    if echo $PYTHON_PATH | grep -q $INSTALL_PATH; then
	    echo "good python version :)"
    else
	cd
	if [ ! -f $SCRIPT ]; then
		curl -LO $DL_LINK
    	fi
    	if [ ! -d $MINICONDA_PATH ]; then
	    	sh $SCRIPT -b -p $INSTALL_PATH"/miniconda3"
	fi
	clear
	echo "Which python:"
	which python
	if grep -q "^export PATH=$MINICONDA_PATH" ~/.zshrc
	then
		echo "export already in .zshrc";
	else
		echo "adding export to .zshrc ...";
		echo "export PATH=$MINICONDA_PATH:\$PATH" >> ~/.zshrc
	fi
	source ~/.zshrc
    fi
}
```

2. Source your `.zshrc` with the following command:

```bash
source ~/.zshrc
```

3. Use the function `set_conda`:

```bash
set_conda
```

When the installation is done rerun the `set_conda` function.


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