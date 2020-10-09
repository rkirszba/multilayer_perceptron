import math
import numpy as np
import matplotlib.pyplot as plt

## revoir question des seeds pour l'initialisation de D et des minibatches,
## checker toutes les divisions par 0 ou log(0)

class FTMultilayerPerceptron():

    def __init__(self, dimensions, max_epoch=1000, batch_size=200, hidden_activation='relu',\
        output_activation='sigmoid', optimizer='gradient_descent', l2_reg=False,\
        dropout_reg=False, weight_initialization='he', alpha=1e-2, lambd=1e-1,\
        keep_probs=None, early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-8,\
        random_state=None, verbose=None):

        self.dimensions_ = dimensions
        self.max_epoch_ = max_epoch
        self.batch_size_ = batch_size
        self.hidden_activation_ = hidden_activation
        self.output_activation_ = output_activation
        self.optimizer_ = optimizer
        self.l2_reg_ = l2_reg
        self.dropout_reg_ = dropout_reg
        self.weight_initialization_ = weight_initialization
        self.alpha_ = alpha
        self.lambda_ = lambd
        self.keep_probs_ = keep_probs
        self.early_stopping_ = early_stopping
        self.beta_ = beta
        self.beta_1_ = beta_1
        self.beta_2_ = beta_2
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

        self.initializers_ = {
            'he': self._he_initialization,
            'xavier': self._xavier_initialization,
            'bengio': self._bengio_initialization
        }

        self.forward_activations_ = {
            'relu': self._relu_forward,
            'tanh': self._tanh_forward,
            'sigmoid': self._sigmoid_forward,
            'softmax': self._softmax_forward
        }
    
        self.backward_activations_ = {
            'relu': self._relu_backward,
            'tanh': self._tanh_backward,
            'sigmoid': self._sigmoid_backward,
            'softmax': self._softmax_backward
        }

        self.optimizers_initializers_ = {
            'momentum': self._momentum_initializer,
            'rmsprop': self._rmsprop_initializer,
            'adam': self._adam_initializer
        }

        self.optimizers_ = {
            'gradient_descent': self._gradient_descent_optimizer,
            'momentum': self._momentum_optimizer,
            'rms_prop': self._rmsprop_optimizer,
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

    def _init_dropout_mask(self):
        self.D_ = []
        self.D_.append(None)
        for layer in range(1, len(self.dimensions_)):
            self.D_.append(np.random.rand(self.dimensions_[layer], self.dimensions_[layer - 1]))
            self.D_[layer] = np.where(self.D_[layer] < self.keep_probs_[layer], 1, 0)



    def _relu_forward(self, layer):
        self.A_.append(np.maximum(0, self.Z_[layer]))

    def _tanh_forward(self, layer):
        self.A_.append(np.tanh(self.Z_[layer]))

    def _sigmoid_forward(self, layer):
        self.A_.append(1 / (1 + np.exp(-self.Z_[layer])))

    def _softmax_forward(self, layer):
        self.A_.append(np.exp(self.Z_[layer]) / np.exp(self.Z_[layer]).sum(axis=0).reshape(1, -1))
    
    def _activation_forward(self, layer):
        if layer < len(self.dimensions_ - 1):
            self.forward_activations_[self.hidden_activation_](layer)
        else:
            self.forward_activations_[self.output_activation_](layer)

    def _linear_forward(self, layer):
        self.Z_.append(np.dot(self.W_[layer], self.A_[layer - 1]))

    def _forward_propagation(self, X):
        self.A_ = []
        self.Z_ = []
        self.Z_.append(None)
        self.A_.append(X)
        if self.dropout_reg_:
            self._init_dropout_mask()
        for layer in range(1, len(self.dimensions_)):
            self._linear_forward(layer)
            self._activation_forward(layer)
            if self.dropout_reg_ and layer != len(self.dimensions_ - 1):
                self.A_[layer] *= self.D_[layer]
                self.A_[layer] /= self.keep_probs_[layer]



    def _relu_backward(self, layer):
        self.dZ_.insert(0, np.where(self.Z_[layer] <= 0, 0, self.dA_[0]))

    def _tanh_backward(self, layer):
        tanh_prime = 1 - np.square(np.tanh(self.Z_[layer]))
        self.dZ_.insert(0, self.dA_[0] * tanh_prime)

    def _sigmoid_backward(self, layer):
        s = 1 / (1 + np.exp(-self.Z_[layer]))
        s_prime = s * (1 - s)
        self.dZ_.insert(0, self.dA_[0] * s_prime)

    def _softmax_backward(self, layerd):
        pass
    
    def _activation_backward(self, layer):
        if layer < len(self.dimensions_ - 1):
            self.backward_activations_[self.hidden_activation_](layer)
        else:
            self.backward_activations_[self.output_activation_](layer)
        
    def _cross_entropy_backward(self, y):
        '''
        Lets be careful, maybe it s not the same with softmax, or with l2_reg
        '''
        self.dA_.insert(0, - (y / self.A_[-1] - (1 - y) / (1 - self.A_[-1])))
    
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
        self._cross_entropy_backward(y)
        for layer in reversed(range(1, len(self.dimensions_))):
            self._activation_backward(layer)
            self._linear_backward(layer)
            if self.l2_reg_:
                self.dW_[0] += (self.lambda_ / m) * self.W_[layer]
            if self.dropout_reg_:
                self.dA_[0] *= self.D_[layer - 1]
                self.dA_[0] /= keep_probs[layer - 1]
            

    def _cross_entropy_cost(self, y):
        y_hat = self.A_[-1]
        return (-1 / y.shape[1]) * np.sum(y * np.log(y_hat + self.epsilon_) + (1 - y) * np.log(1 - y_hat + self.epsilon_))
    
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

    def _random_mini_batches(self, X, y, seed=0):
        m = X.shape[1]
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        y_shuffled = y[:, permutation].reshape(-1, m)
        nb_batches = m / self.batch_size_
        X_batches = []
        y_batches = []
        for i in nb_batches:
            X_batches.append(X_shuffled[:, i * self.batch_size_ : (i + 1) * self.batch_size_])
            y_batches.append(y_shuffled[:, i * self.batch_size_ : (i + 1) * self.batch_size_])
        if m % self.batch_size_ != 0:
            X_batches.append(X_shuffled[:, nb_batches * self.batch_size_ :])
            y_batches.append(y_shuffled[:, nb_batches * self.batch_size_ :])
        return X_batches, y_batches
    

    def _gradient_descent_optimizer(self):
        for i in range(1, len(self.W_)):    
            self.W_[i] -= self.alpha_ * self.dW_[i]
            self.b_[i] -= self.alpha_ * self.db_[i]

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
            self.b_[layer] -= self.alpha_ * (self.db_[layer] / (np.sqrt(self.sdW_[layer]) + self.epsilon_))

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
            v_corrected_dW.append(self.vdW_[layer] / (1 / math.pow(self.beta_1_, self.adam_iter_)))
            v_corrected_db.append(self.vdb_[layer] / (1 / math.pow(self.beta_1_, self.adam_iter_)))
            self.sdW_[layer] = self.beta_2_ * self.sdW_[layer] + (1 - self.beta_2_) * np.square(self.dW_[layer])
            self.sdb_[layer] = self.beta_2_ * self.sdb_[layer] + (1 - self.beta_2_) * np.square(self.db_[layer])
            s_corrected_dW.append(self.sdW_[layer] / (1 / math.pow(self.beta_2_, self.adam_iter_)))
            s_corrected_db.append(self.sdb_[layer] / (1 / math.pow(self.beta_2_, self.adam_iter_)))
            self.W_[layer] -= self.alpha_ * (v_corrected_dW[layer] / (np.sqrt(s_corrected_dW[layer]) + self.epsilon_))
            self.b_[layer] -= self.alpha_ * (v_corrected_db[layer] / (np.sqrt(s_corrected_db[layer]) + self.epsilon_))

    def _update_parameters(self):
        self.optimizers_[self.optimizer_]

    def _momentum_initializer(self):
        self.vdW_.append(None)
        self.vdb_.append(None)
        for layer in range(1, len(self.dimensions_)):
            self.vdW.append(np.zeros(self.W_[layer].shape))
            self.vdb.append(np.zeros(self.b_[layer].shape))

    def _rmsprop_initializer(self):
        self.sdW_.append(None)
        self.sdb_.append(None)
        for layer in range(1, len(self.dimensions_)):
            self.sdW.append(np.zeros(self.W_[layer].shape))
            self.sdb.append(np.zeros(self.b_[layer].shape))

    def _adam_initializer(self):
        self.vdW_.append(None)
        self.vdb_.append(None)
        self.sdW_.append(None)
        self.sdb_.append(None)
        for layer in range(1, len(self.dimensions_)):
            self.vdW.append(np.zeros(self.W_[layer].shape))
            self.vdb.append(np.zeros(self.b_[layer].shape))
            self.sdW.append(np.zeros(self.W_[layer].shape))
            self.sdb.append(np.zeros(self.b_[layer].shape))


    def fit(self, X, y, X_dev=None, y_dev=None):
        if self.random_state_:
            np.random.seed(self.random_state_)
        self._init_parameters()
        if self.optimizer_ in self.optimizers_initializers_.keys():
            self.optimizers_initializers_[self.optimizer_]()
        for epoch in range(self.max_epoch_):
            X_batches, y_batches = self._random_mini_batches(X, y, seed=epoch)
            cost_train = 0
            if X_dev and y_dev:
                self._forward_propagation(X_dev)
                self.costs_dev_.append(self._compute_cost(y_dev))
            for X_batch, y_batch in zip(X_batches, y_batches):
                self._forward_propagation(X_batch)
                cost_train += self._compute_cost(y_batch) * X_batch.shape[1]
                self._backward_propagation(y_batch)
                self._update_parameters()
            self.costs_train_.append(cost_train / X.shape[1])
            if self.verbose_ and epoch % self.verbose_ == 0:
                message = 'epoch {}/{} - loss: {}'.format(epoch, self.max_epoch_, self.costs_train_[-1])
                if X_dev and y_dev:
                    message += ' - val_loss: {}'.format(self.costs_dev_[-1])
                print(message)
            if self.early_stopping_:
                pass
                
    def print_learning_curve(costs_train=True, costs_dev=False):
        if costs_train:
            plt.plot(self.costs_train_)
        if costs_dev:
            plt.plot(self.costs_dev_)
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.show()

    def get_costs():
        return self.costs_train_, self.costs_dev_



