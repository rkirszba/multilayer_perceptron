import numpy as np

## revoir question des seeds pour l'initialisation de D et des minibatches

class FTMultilayerPerceptron():

    def __init__(dimensions, max_epoch=1000, hidden_activation='relu', output_activation='sigmoid',\
        optimizer='gradient_descent', l2_reg=False, dropout_reg=False, weight_initialization='he',\
        alpha=1e-2, lambd=0., keep_probs=None, early_stopping=False, beta_1=0.9,\
        beta_2=0.999, epsilon=1e-8, random_state=None, verbose=False):

        self.dimensions_ = dimensions
        self.max_epoch_ = max_epoch
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
        self.beta_1_ = beta1
        self.beta_2_ = beta_2
        self.epsilon_ = epsilon
        self.random_state_ = random_state_
        self.verbose_ = verbose

        self.costs_train_ = []
        self.costs_dev_ = []

        self.W_ = []
        self.b_ = []
        self.A_ = []
        self.Z_ = []
        self.D_ = []
        self.dW_ = []
        self.db_ = []
        self.dA_ = []
        self.dZ_ = []

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
    
    def _he_initialization(layer):
        return np.random.randn(self.dimensions_[layer], self.dimensions_[layer - 1])\
            * np.sqrt(2 / self.dimensions_[layer - 1]))

    def _xavier_initialization(layer):
        return np.random.randn(self.dimensions_[layer], self.dimensions_[layer - 1])\
            * np.sqrt(1 / self.dimensions_[layer - 1]))

    def _bengio_initialization(layer):
        return np.random.randn(self.dimensions_[layer], self.dimensions_[layer - 1])\
            * np.sqrt(2 / (self.dimensions_[layer - 1] + self.dimensions_[layer])))

    def _init_parameters(self):
        if self.random_state_:
            np.random.seed(self.random_state_)
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
            self.D_[layer] = np.where(self.D_[layer] < self.keep_probs[layer], 1, 0)

    @staticmethod
    def _relu_forward(Z):
        return np.maximum(0, Z)

    @staticmethod
    def _tanh_forward(Z):
        return np.tanh(Z)

    @staticmethod
    def _sigmoid_forward(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def _softmax_forward(Z):
        pass
    
    @staticmethod
    def _activation_forward(Z, activation):
        return activation(Z)

    @staticmethod
    def _linear_forward(W, A_prev):
        return np.dot(W, A_prev)

    def _get_activation_function(self, layer):
        if layer < len(self.dimensions_ - 1):
            return self.forward_activations_[self.hidden_activation_]
        else:
            return self.forward_activations_[self.output_activation_]

    def _forward_propagation(self, X):
        self.Z_.append(None)
        self.A_.append(X)
        if self.dropout_reg:
            self.D_ = self._init_dropout_mask()
        for layer in range(1, len(self.dimensions_)):
            self.Z_.append._linear_forward(self.W_[layer], self.A_[layer - 1])
            self.A_.append._activation_forward(self.Z_[layer], self._get_activation_function(layer))
            if self.dropout_reg and layer != len(self.dimensions_ - 1):
                self.A_[layer] *= self.D_[layer]
                self.A[layer] /= self.keep_probs_[layer]

    @staticmethod
    def _relu_backward(dA, Z):
        return np.where(Z <= 0, 0, dA)

    @staticmethod
    def _tanh_backward(dA, Z):

    @staticmethod
    def _sigmoid_backward(dA, Z):
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ

    @staticmethod
    def _softmax_backward(dA, Z):
        pass
    
    @staticmethod
    def _activation_backward(dA, Z, activation):
        return activation(dA, Z)
    
    @staticmethod
    def _cost_backward(A, y):
        '''
        Lets be careful, maybe it s not the same with softmax, or with l2_reg
        '''
        return np.divide(y, A) - np.divide(1 - y, 1 - A)
    
    @staticmethod
    def _linear_backward(A_prev, W, b, dZ):
        m = A_prev.shape[1]
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dW, db, dA_prev


    def _backward_propagation(self, y):
        self.dA_ = []
        self.dZ_ = []
        self.dW_ = []
        self.db_ = []
        self.dA_.insert(0, self._cost_backward(self.A_[-1], y))
        for layer in reversed(range(1, len(self.dimensions_)))
            self.dZ_.insert(0, self._activation_backward(self.dA_[0], self.Z_[layer], self._get_activation_function(layer)))
            dW, db, dA_prev = self._linear_backward(self.A_[layer - 1], self.W_[layer], self.b_[layer], dZ[0])
            if self.dropout_reg_:
                pass
            if self.l2_reg_:
                pass
            self.dA_.insert(0, dA_prev)
            self.dW_.insert(0, dW)
            self.db_.insert(0, db)
            

    def _cross_entropy_cost(self, y):
        y_hat = self.A_[-1]
        return (-1 / y.shape[1]) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    def _l2_reg_cost(self, m):
        l2_reg_cost = 0
        for W in self.W_[1:]:
            l2_reg_cost += np.squeeze(np.sum(np.square(W)))
        return (self.lambda_ / (2 * m)) * l2_reg_cost
    
    def _compute_cost(self, y):
        cost = self._cross_entropy_cost(y)
        if self.l2_reg_:
            cost += self._l2_reg_cost(y.shape[1])
'''
    def fit(X, y, X_dev=None):

        self._init_parameters()
        for epoch in range(self.max_epoch_):
            X_shuffle = None
            Y_shuffle = None
            nb_batches = None
            cost = 
            for i in range(nb_batches):
                self._forward_propagation(X)
                cost 
'''



