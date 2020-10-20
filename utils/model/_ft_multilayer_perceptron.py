import math
import numpy as np
import matplotlib.pyplot as plt


class FTMultilayerPerceptron():

    def __init__(self, dimensions, max_epoch=1000, batch_size=200, hidden_activation='relu',\
        optimizer='gradient_descent', l2_reg=False, dropout_reg=False, weight_initialization='he',\
        alpha=1e-2, lambd=1e-1, keep_probs=None, early_stopping=False, decay_rate=None,\
        alpha_0=0.2, beta=0.9, beta_1=0.9, beta_2=0.999, patience=50, epsilon=1e-8,\
        random_state=None, verbose=None):

        self.dimensions_ = dimensions
        self.max_epoch_ = max_epoch
        self.batch_size_ = batch_size
        self.hidden_activation_ = hidden_activation
        self.optimizer_ = optimizer
        self.l2_reg_ = l2_reg
        self.dropout_reg_ = dropout_reg
        self.weight_initialization_ = weight_initialization
        self.alpha_ = alpha
        self.lambda_ = lambd
        self.keep_probs_ = keep_probs
        self.early_stopping_ = early_stopping
        self.decay_rate_ = decay_rate
        self.alpha_0_ = alpha_0
        self.beta_ = beta
        self.beta_1_ = beta_1
        self.beta_2_ = beta_2
        self.patience_ = patience
        self.epsilon_ = epsilon
        self.random_state_ = random_state
        self.verbose_ = verbose

        self.W_ = []
        self.b_ = []
        self.A_ = []
        self.Z_ = []
        self.dW_ = []
        self.db_ = []
        self.dA_ = []
        self.dZ_ = []

        self.D_ = []

        self.vdW_ = []
        self.vdb_ = []
        self.sdW_ = []
        self.sdb_ = []
        self.adam_iter_ = 0

        self.costs_train_ = []
        self.costs_dev_ = []

        self.no_improv_ = 0

        self.initializers_ = {
            'he': self._he_initialization,
            'xavier': self._xavier_initialization,
            'bengio': self._bengio_initialization
        }

        self.forward_hidden_activations_ = {
            'relu': self._relu_forward,
            'lrelu': self._lrelu_forward,
            'tanh': self._tanh_forward,
            'sigmoid': self._sigmoid_forward
        }
    
        self.backward_hidden_activations_ = {
            'relu': self._relu_backward,
            'lrelu': self._lrelu_backward,
            'tanh': self._tanh_backward,
            'sigmoid': self._sigmoid_backward
        }

        self.optimizers_initializers_ = {
            'momentum': self._momentum_initialization,
            'rmsprop': self._rmsprop_initialization,
            'adam': self._adam_initialization
        }

        self.optimizers_ = {
            'gradient_descent': self._gradient_descent_optimizer,
            'momentum': self._momentum_optimizer,
            'rmsprop': self._rmsprop_optimizer,
            'adam': self._adam_optimizer
        }



    def _he_initialization(self, layer):
        return np.random.randn(self.dimensions_[layer], self.dimensions_[layer - 1])\
            * np.sqrt(2 / self.dimensions_[layer - 1])

    def _xavier_initialization(self, layer):
        return np.random.randn(self.dimensions_[layer], self.dimensions_[layer - 1])\
            * np.sqrt(1 / self.dimensions_[layer - 1])

    def _bengio_initialization(self, layer):
        return np.random.randn(self.dimensions_[layer], self.dimensions_[layer - 1])\
            * np.sqrt(2 / (self.dimensions_[layer - 1] + self.dimensions_[layer]))

    def _init_parameters(self):
        self.W_.append(None)
        self.b_.append(None)
        for layer in range(1, len(self.dimensions_)):
            self.W_.append(self.initializers_[self.weight_initialization_](layer))
            self.b_.append(np.zeros((self.dimensions_[layer], 1)))

    def _init_dropout_mask(self, m):
        self.D_ = []
        self.D_.append(None)
        for layer in range(1, len(self.dimensions_) - 1):
            self.D_.append(np.random.rand(self.dimensions_[layer], m))
            self.D_[layer] = np.where(self.D_[layer] < self.keep_probs_[layer], 1, 0)

    def _momentum_initialization(self):
        self.vdW_.append(None)
        self.vdb_.append(None)
        for layer in range(1, len(self.dimensions_)):
            self.vdW_.append(np.zeros(self.W_[layer].shape))
            self.vdb_.append(np.zeros(self.b_[layer].shape))

    def _rmsprop_initialization(self):
        self.sdW_.append(None)
        self.sdb_.append(None)
        for layer in range(1, len(self.dimensions_)):
            self.sdW_.append(np.zeros(self.W_[layer].shape))
            self.sdb_.append(np.zeros(self.b_[layer].shape))

    def _adam_initialization(self):
        self.vdW_.append(None)
        self.vdb_.append(None)
        self.sdW_.append(None)
        self.sdb_.append(None)
        for layer in range(1, len(self.dimensions_)):
            self.vdW_.append(np.zeros(self.W_[layer].shape))
            self.vdb_.append(np.zeros(self.b_[layer].shape))
            self.sdW_.append(np.zeros(self.W_[layer].shape))
            self.sdb_.append(np.zeros(self.b_[layer].shape))

    def _init_all(self):
        self._init_parameters()
        if self.optimizer_ in self.optimizers_initializers_.keys():
            self.optimizers_initializers_[self.optimizer_]()

    def _relu_forward(self, layer):
        self.A_.append(np.maximum(0, self.Z_[layer]))

    def _lrelu_forward(self, layer):
        self.A_.append(np.maximum(0.1 * self.Z_[layer], self.Z_[layer]))

    def _tanh_forward(self, layer):
        self.A_.append(np.tanh(self.Z_[layer]))

    def _sigmoid_forward(self, layer):
        self.A_.append(1 / (1 + np.exp(-self.Z_[layer])))

    def _softmax_forward(self, layer):
        self.A_.append(np.exp(self.Z_[layer]) / np.exp(self.Z_[layer]).sum(axis=0, keepdims=True))
    
    def _activation_forward(self, layer):
        if layer < len(self.dimensions_) - 1:
            self.forward_hidden_activations_[self.hidden_activation_](layer)
        else:
            self._softmax_forward(layer)

    def _linear_forward(self, layer):
        self.Z_.append(np.dot(self.W_[layer], self.A_[layer - 1]) + self.b_[layer])

    def _forward_propagation(self, X, purpose='train'):
        self.A_ = []
        self.Z_ = []
        self.Z_.append(None)
        self.A_.append(X)
        if self.dropout_reg_ and purpose == 'train':
            self._init_dropout_mask(X.shape[1])
        for layer in range(1, len(self.dimensions_)):
            self._linear_forward(layer)
            self._activation_forward(layer)
            if purpose == 'train' and self.dropout_reg_ and layer != len(self.dimensions_) - 1:
                self.A_[layer] *= self.D_[layer]
                self.A_[layer] /= self.keep_probs_[layer]


    def _relu_backward(self, layer):
        self.dZ_.insert(0, np.where(self.Z_[layer] <= 0, 0, self.dA_[0]))

    def _lrelu_backward(self, layer):
        self.dZ_.insert(0, np.where(self.Z_[layer] <= 0, 0.1 * self.dA_[0], self.dA_[0]))

    def _tanh_backward(self, layer):
        tanh_prime = 1 - np.square(np.tanh(self.Z_[layer]))
        self.dZ_.insert(0, self.dA_[0] * tanh_prime)

    def _sigmoid_backward(self, layer):
        s = 1 / (1 + np.exp(-self.Z_[layer]))
        sigmoid_prime = s * (1 - s)
        self.dZ_.insert(0, self.dA_[0] * sigmoid_prime)

    def _softmax_backward(self, layer, y):
        self.dZ_.insert(0, self.A_[layer] - y)
    
    def _activation_backward(self, layer, y):
        if layer < len(self.dimensions_) - 1:
            self.backward_hidden_activations_[self.hidden_activation_](layer)
        else:
            self._softmax_backward(layer, y)
        
    def _linear_backward(self, layer):
        m = self.A_[layer - 1].shape[1]
        self.dW_.insert(0, (1 / m) * np.dot(self.dZ_[0], self.A_[layer - 1].T))
        self.db_.insert(0, (1 / m) * np.sum(self.dZ_[0], axis=1, keepdims=True))
        self.dA_.insert(0, np.dot(self.W_[layer].T, self.dZ_[0]))

    def _backward_propagation(self, y):
        self.dA_ = []
        self.dZ_ = []
        self.dW_ = []
        self.db_ = []
        m = y.shape[1]
        for layer in reversed(range(1, len(self.dimensions_))):
            self._activation_backward(layer, y)
            self._linear_backward(layer)
            if self.l2_reg_:
                self.dW_[0] += (self.lambda_ / m) * self.W_[layer]
            if self.dropout_reg_ and layer != 1:
                self.dA_[0] *= self.D_[layer - 1]
                self.dA_[0] /= self.keep_probs_[layer - 1]
        self.dW_.insert(0, None)
        self.db_.insert(0, None)
            
    def _cross_entropy_cost_alter(self, y):
        y_hat = self.A_[-1]
        return np.squeeze((-1 / y.shape[1]) * np.sum(y * np.log(y_hat + self.epsilon_)\
            + (1 - y) * np.log(1 - y_hat + self.epsilon_)))

    def _cross_entropy_cost(self, y):
        y_hat = self.A_[-1]
        return np.squeeze((-1 / y.shape[1]) * np.sum(y * np.log(y_hat + self.epsilon_)))
        


    def _l2_reg_cost(self, m):
        l2_reg_cost = 0
        for W in self.W_[1:]:
            l2_reg_cost += np.squeeze(np.sum(np.square(W)))
        return (self.lambda_ / (2 * m)) * l2_reg_cost
    
    def _compute_cost(self, y):
        cost = self._cross_entropy_cost(y)
        if self.l2_reg_:
            cost += self._l2_reg_cost(y.shape[1])
        return cost

    def _random_mini_batches(self, X, y):
        m = X.shape[1]
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation].reshape(-1, m)
        y_shuffled = y[:, permutation].reshape(-1, m)
        nb_batches = m // self.batch_size_
        X_batches = []
        y_batches = []
        for i in range(nb_batches):
            X_batches.append(X_shuffled[:, i * self.batch_size_ : (i + 1) * self.batch_size_])
            y_batches.append(y_shuffled[:, i * self.batch_size_ : (i + 1) * self.batch_size_])
        if m % self.batch_size_ != 0:
            X_batches.append(X_shuffled[:, nb_batches * self.batch_size_ :])
            y_batches.append(y_shuffled[:, nb_batches * self.batch_size_ :])
        return X_batches, y_batches
    

    def _gradient_descent_optimizer(self):
        for i in range(1, len(self.W_)):
            tmpW = self.W_[i]
            self.W_[i] = self.W_[i] - self.alpha_ * self.dW_[i]
            self.b_[i] = self.b_[i] - self.alpha_ * self.db_[i]


    def _momentum_optimizer(self):
        for layer in range(1, len(self.dimensions_)):
            self.vdW_[layer] = self.beta_ * self.vdW_[layer] + (1 - self.beta_) * self.dW_[layer]
            self.vdb_[layer] = self.beta_ * self.vdb_[layer] + (1 - self.beta_) * self.db_[layer]
            self.W_[layer] -= self.alpha_ * self.vdW_[layer]
            self.b_[layer] -= self.alpha_ * self.vdb_[layer]

    def _rmsprop_optimizer(self):
        for layer in range(1, len(self.dimensions_)):
            self.sdW_[layer] = self.beta_ * self.sdW_[layer] + (1 - self.beta_) * np.square(self.dW_[layer])
            self.sdb_[layer] = self.beta_ * self.sdb_[layer] + (1 - self.beta_) * np.square(self.db_[layer])
            self.W_[layer] -= self.alpha_ * (self.dW_[layer] / (np.sqrt(self.sdW_[layer]) + self.epsilon_))
            self.b_[layer] -= self.alpha_ * (self.db_[layer] / (np.sqrt(self.sdb_[layer]) + self.epsilon_))

    def _adam_optimizer(self):
        v_corrected_dW = []
        v_corrected_db = []
        s_corrected_dW = []
        s_corrected_db = []
        v_corrected_dW.append(None)
        v_corrected_db.append(None)
        s_corrected_dW.append(None)
        s_corrected_db.append(None)
        self.adam_iter_ += 1
        for layer in range(1, len(self.dimensions_)):
            self.vdW_[layer] = self.beta_1_ * self.vdW_[layer] + (1 - self.beta_1_) * self.dW_[layer]
            self.vdb_[layer] = self.beta_1_ * self.vdb_[layer] + (1 - self.beta_1_) * self.db_[layer]
            v_corrected_dW.append(self.vdW_[layer] / (1 - math.pow(self.beta_1_, self.adam_iter_)))
            v_corrected_db.append(self.vdb_[layer] / (1 - math.pow(self.beta_1_, self.adam_iter_)))
            self.sdW_[layer] = self.beta_2_ * self.sdW_[layer] + (1 - self.beta_2_) * np.square(self.dW_[layer])
            self.sdb_[layer] = self.beta_2_ * self.sdb_[layer] + (1 - self.beta_2_) * np.square(self.db_[layer])
            s_corrected_dW.append(self.sdW_[layer] / (1 - math.pow(self.beta_2_, self.adam_iter_)))
            s_corrected_db.append(self.sdb_[layer] / (1 - math.pow(self.beta_2_, self.adam_iter_)))
            self.W_[layer] -= self.alpha_ * (v_corrected_dW[layer] / (np.sqrt(s_corrected_dW[layer]) + self.epsilon_))
            self.b_[layer] -= self.alpha_ * (v_corrected_db[layer] / (np.sqrt(s_corrected_db[layer]) + self.epsilon_))

    def _update_parameters(self):
        self.optimizers_[self.optimizer_]()



    def _stop_learning(self):
        if self.costs_dev_[-1] >= self.costs_dev_[-2]:
            self.no_improv_ += 1
        else:
            self.no_improv_ =0
        return self.no_improv_ == self.patience_

    
    def _verbose_message(self, epoch, end=False):
        message = ''
        if end == True:
            message += 'End of training:\n'
        message += 'epoch {}/{} - loss: {}'.format(epoch, self.max_epoch_, self.costs_train_[-1])
        if len(self.costs_dev_) > 0:
            message += ' - val_loss: {}'.format(self.costs_dev_[-1])
        return message

    def _update_alpha(self, epoch):
        if self.decay_rate_ is not None:
            self.alpha_ = (1 / (1 + self.decay_rate_ * epoch)) * self.alpha_0_

    def fit(self, X, y, X_dev=None, y_dev=None):
        if self.random_state_ is not None:
            np.random.seed(self.random_state_)
        self._init_all()
        final_epoch = self.max_epoch_
        for epoch in range(self.max_epoch_):
            self._update_alpha(epoch)
            X_batches, y_batches = self._random_mini_batches(X, y)
            cost_train = 0
            for X_batch, y_batch in zip(X_batches, y_batches):
                self._forward_propagation(X_batch)
                cost_train += self._compute_cost(y_batch) * X_batch.shape[1]
                self._backward_propagation(y_batch)
                self._update_parameters()
            self.costs_train_.append(cost_train / X.shape[1])
            if X_dev is not None and y_dev is not None:
                self._forward_propagation(X_dev, purpose='test')
                self.costs_dev_.append(self._compute_cost(y_dev))
            if self.verbose_ and epoch % self.verbose_ == 0:
                print(self._verbose_message(epoch))
            if epoch > 0 and self.early_stopping_:
                if self._stop_learning():
                    final_epoch = epoch
                    break
        if self.verbose_:
            print(self._verbose_message(final_epoch, end=True))
        return self

    def predict_probas(self, X):
        self._forward_propagation(X, purpose='test')
        return self.A_[-1]

    def predict(self, X):
        self._forward_propagation(X, purpose='test')
        return np.argmax(self.A_[-1], axis=0).reshape(1, -1)

                
    def plot_learning_curve(self, costs_train=True, costs_dev=False):
        if costs_train:
            plt.plot(self.costs_train_, label='Train set', color='blue')
        if costs_dev:
            plt.plot(self.costs_dev_, label='Dev set', color='orange')
        plt.legend()
        plt.ylabel('Cost')
        plt.xlabel('Epochs')
        plt.show()

    def get_costs_history(self):
        return self.costs_train_, self.costs_dev_



