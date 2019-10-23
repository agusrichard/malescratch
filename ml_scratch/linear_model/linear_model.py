import numpy as np 
import math
import MinMaxScaler, StandardScaler, make_batch_index, r2_score, to_categorical

# ======================================================================================================================

class l1_regularization():

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, 1)

    def grad(self, w):
        return self.alpha * np.sign(w)


# ======================================================================================================================

class l2_regularization():

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w, 2)

    def grad(self, w):
        return self.alpha * np.sum(w ** 2) * w


# ======================================================================================================================

class l1_l2_regularization():

    def __init__(self, alpha, l1_ratio):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        l1_contrib = self.l1_ratio * np.linalg.norm(w, 1)
        l2_contrib = (1 - self.l1_ratio) * np.linalg.norm(w, 2)

        return self.alpha * (l1_contrib + l2_contrib)

    def grad(self, w):
        l1_contrib = self.l1_ratio * np.sign(w)
        l2_contrib = (1 - self.l1_ratio) * np.sum(w ** 2) * w

        return self.alpha * (l1_contrib + l2_contrib)


# ======================================================================================================================

class Regression(object):

    def __init__(self):
        pass

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self._weights)

    def score(self, X, y):
        predictions = self.predict(X)
        labels = y

        return r2_score(predictions, labels)


# ======================================================================================================================

class LinearRegression(Regression):

    def __init__(self, regularization=False, alpha=0.0001):
        self.regularization = regularization
        self.alpha = alpha

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        # Calculate weights by least squares (using Moore-Penrose pseudoinverse)
        if self.regularization:
            regularization_mat = np.eye(X.shape[1])
            regularization_mat[0, 0] = 0
            mat = X.T.dot(X) + regularization_mat
            U, S, V = np.linalg.svd(mat)
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self._weights = X_sq_reg_inv.dot(X.T).dot(y)
        else:
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            X_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self._weights = X_sq_reg_inv.dot(X.T).dot(y)
        
        self.weights_ = self._weights.ravel()[1:]
        self.bias_ = self._weights[0]
        return self


# ======================================================================================================================

class BatchGDRegressor(Regression):

    def __init__(self, num_batch=None, iterations=10000, learning_rate=0.001, penalty=None, alpha=0.0001, l1_ratio=0.15):
        self.num_batch = num_batch
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if penalty == 'l1':
            self._regularization = l1_regularization(alpha=self.alpha)
        elif penalty == 'l2':
            self._regularization = l2_regularization(alpha=self.alpha)
        elif penalty == 'elasticnet':
            self._regularization = l1_l2_regularization(alpha=self.alpha, l1_ratio=self.l1_ratio)
        elif penalty == None:
            self._regularization = lambda x: 0
            self._regularization.grad = lambda x: 0

    def _weights_init(self, n_features):
        lim = 1 / math.sqrt(n_features)
        self._weights = np.random.uniform(-lim, lim, size=(n_features, 1))
        
    def fit(self, X, y):
        if self.num_batch == None:
            self.num_batch = int(X.shape[0] / 20)
        X = np.c_[np.ones(X.shape[0]), X]
        y = y.reshape(-1, 1)
        self.loss_vals_ = []
        self._weights_init(X.shape[1])
        percentage = 0 

        for i in range(self.iterations):
            batch_index = make_batch_index(sample_size=X.shape[0], num_batch=self.num_batch, 
                                            size=X.shape[0], shuffle=True, random_state=42)
 
            for batch in batch_index:
                batch_X = X[batch]
                batch_y = y[batch]

                y_pred = batch_X.dot(self._weights)
                loss_val = 0.5 * np.mean((y_pred - batch_y)**2) + self._regularization(self._weights)
                self.loss_vals_.append(loss_val)
                gradient = batch_X.T.dot(y_pred - batch_y) + self._regularization.grad(self._weights)
                self._weights -= (self.learning_rate * gradient)
                
            if i % (self.iterations/100) == 0:
                print("\rProcess {}%, loss_val: {}".format(percentage, loss_val), end='')
                percentage += 1
        print("\rProcess 100%, loss_val: {}".format(loss_val), end='')
        
        self.weights_ = self._weights.ravel()[1:]
        self.bias_ = self._weights[0]
        self.loss_vals_ = np.array(self.loss_vals_)

        return self


# ======================================================================================================================

class RidgeRegression(BatchGDRegressor):

    def __init__(self, num_batch=None, iterations=10000, learning_rate=0.001, alpha=0.0001):
        super().__init__(num_batch=num_batch,
                         iterations=iterations,
                         learning_rate=learning_rate,
                         penalty='l2',
                         alpha=alpha,
                         l1_ratio=None)
        
    def fit(self, X, y):
        super().fit(X, y)

        return self


# ======================================================================================================================

class LassoRegression(BatchGDRegressor):

    def __init__(self, num_batch=None, iterations=10000, learning_rate=0.001, alpha=0.0001):
        super().__init__(num_batch=num_batch,
                         iterations=iterations,
                         learning_rate=learning_rate,
                         penalty='l1',
                         alpha=alpha,
                         l1_ratio=None)
        
    def fit(self, X, y):
        super().fit(X, y)

        return self


# ======================================================================================================================

class ElasticNetRegression(BatchGDRegressor):

    def __init__(self, num_batch=None, iterations=10000, learning_rate=0.001, alpha=0.0001, l1_ratio=0.15):
        super().__init__(num_batch=num_batch,
                         iterations=iterations,
                         learning_rate=learning_rate,
                         penalty='elasticnet',
                         alpha=alpha,
                         l1_ratio=0.15)
        
    def fit(self, X, y):
        super().fit(X, y)

        return self


# ======================================================================================================================

class StochasticGDRegressor(BatchGDRegressor):

    def __init__(self, iterations=1000, learning_rate=0.001, penalty=None, alpha=0.0001, l1_ratio=0.15):
        super().__init__(num_batch=None,
                         iterations=iterations,
                         learning_rate=learning_rate,
                         penalty=penalty,
                         alpha=alpha,
                         l1_ratio=l1_ratio)
        
    def fit(self, X, y):
        self.num_batch = X.shape[0]
        super().fit(X, y)

        return self


# ======================================================================================================================

class LogisticRegression():

    def __init__(self, num_batch=None, iterations=1000, learning_rate=0.001, penalty=None, alpha=0.0001, l1_ratio=0.15):
        self.num_batch = num_batch
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if penalty == 'l1':
            self._regularization = l1_regularization(alpha=self.alpha)
        elif penalty == 'l2':
            self._regularization = l2_regularization(alpha=self.alpha)
        elif penalty == 'elasticnet':
            self._regularization = l1_l2_regularization(alpha=self.alpha, l1_ratio=self.l1_ratio)
        elif penalty == None:
            self._regularization = lambda x: 0
            self._regularization.grad = lambda x: 0

    def _weights_init(self, n_features, n_labels):
        lim = 1 / math.sqrt(n_features)
        self._weights = np.random.uniform(-lim, lim, size=(n_features, n_labels))

    def _calc_sigmoid(self, X, weights):
        z = X.dot(weights)
        return 1/(1+np.exp(-z))
    
    def _calc_gradient(self, X, y_pred, y_train):
        result = X.T.dot(y_pred - y_train)
        return result

    def _calc_loss(self, y_pred, y_train):
        return -(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred))

    def fit(self, X, y):
        if self.num_batch == None:
            self.num_batch = int(X.shape[0] / 20)
        X = np.c_[np.ones(X.shape[0]), X]
        y = to_categorical(y)
        self.loss_vals_ = []
        self._weights_init(X.shape[1], y.shape[1])
        percentage = 0 

        for i in range(self.iterations):
            batch_index = make_batch_index(sample_size=X.shape[0], num_batch=self.num_batch, 
                                            size=X.shape[0], shuffle=True, random_state=42)
            
            for batch in batch_index:
                batch_X = X[batch]
                batch_y = y[batch]

                y_pred = self._calc_sigmoid(batch_X, self._weights)
                loss_val = self._calc_loss(y_pred, batch_y) + self._regularization(self._weights)
                self.loss_vals_.append(loss_val)
                gradient = self._calc_gradient(batch_X, y_pred, batch_y) + self._regularization(self._weights)
                self._weights -= (self.learning_rate * gradient)

            if i % (self.iterations/100) == 0:
                print("\rProcess {}%".format(percentage), end='')
                percentage += 1
        print("\rProcess 100%", end='')
        
        self.loss_vals_ = np.array(self.loss_vals_)

        return self

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        predict_proba = self._calc_sigmoid(X, self._weights)

        self._predictions = np.argmax(predict_proba, 1)
        return self._predictions, predict_proba

    def score(self, X, y):
        try:
            predictions = self._predictions
        except Exception:
            predictions, _ = self.predict(X)
        labels = y

        return accuracy_score(predictions, labels)


# ======================================================================================================================

class SekardayuHanaPradiani():

    def __init__(self, activation='sigmoid', num_batch=None, iterations=1000, learning_rate=0.001, penalty=None, alpha=0.0001, l1_ratio=0.15):
        self.activation = activation
        self.num_batch = num_batch
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        if penalty == 'l1':
            self._regularization = l1_regularization(alpha=self.alpha)
        elif penalty == 'l2':
            self._regularization = l2_regularization(alpha=self.alpha)
        elif penalty == 'elasticnet':
            self._regularization = l1_l2_regularization(alpha=self.alpha, l1_ratio=self.l1_ratio)
        elif penalty == None:
            self._regularization = lambda x: 0
            self._regularization.grad = lambda x: 0

    def _weights_init(self, n_features, n_labels):
        lim = 1 / math.sqrt(n_features)
        self._weights = np.random.uniform(-lim, lim, size=(n_features, n_labels))

    def _activation_sigmoid(self, X, weights):
        z = X.dot(weights)
        return 1/(1+np.exp(-z))

    def _activation_softmax(self, X, weights):
        z = X.dot(weights)
        summa = np.sum(np.exp(z))
        return np.exp(z) / summa

    def _activation_tanh(self, X, weights):
        z = X.dot(weights)
        return (np.tanh(z) + 1) / 2

    def _activation_relu(self, X, weights):
        z = X.dot(weights)
        return np.maximum(0, z)
    
    def _calc_gradient(self, X, y_pred, y_train):
        result = X.T.dot(y_pred - y_train)
        return result

    def _calc_loss(self, y_pred, y_train):
        return np.mean(-(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred)))

    def fit(self, X, y):
        if self.num_batch == None:
            self.num_batch = int(X.shape[0] / 20)
        X = np.c_[np.ones(X.shape[0]), X]
        y = to_categorical(y)
        self.loss_vals_ = []
        self._weights_init(X.shape[1], y.shape[1])
        percentage = 0 

        for i in range(self.iterations):
            batch_index = make_batch_index(sample_size=X.shape[0], num_batch=self.num_batch, 
                                            size=X.shape[0], shuffle=True, random_state=42)
            
            for batch in batch_index:
                batch_X = X[batch]
                batch_y = y[batch]

                if self.activation == 'sigmoid':
                    y_pred = self._activation_sigmoid(batch_X, self._weights)
                elif self.activation == 'softmax':
                    y_pred = self._activation_softmax(batch_X, self._weights)
                elif self.activation == 'tanh':
                    y_pred = self._activation_tanh(batch_X, self._weights)
                elif self.activation == 'relu':
                    y_pred = self._activation_relu(batch_X, self._weights)
                else:
                    raise ValueError("Unknown activation hyperparameter")
                loss_val = self._calc_loss(y_pred, batch_y) + self._regularization(self._weights)
                self.loss_vals_.append(loss_val)
                gradient = self._calc_gradient(batch_X, y_pred, batch_y) + self._regularization(self._weights)
                self._weights -= (self.learning_rate * gradient)

            if i % (self.iterations/100) == 0:
                print("\rProcess {}%".format(percentage), end='')
                percentage += 1
        print("\rProcess 100%", end='')
        
        self.loss_vals_ = np.array(self.loss_vals_)

        return self

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        if self.activation == 'sigmoid':
            predict_proba = self._activation_sigmoid(X, self._weights)
        elif self.activation == 'softmax':
            predict_proba = self._activation_softmax(X, self._weights)
        elif self.activation == 'tanh':
            predict_proba = self._activation_tanh(X, self._weights)
        elif self.activation == 'relu':
            predict_proba = self._activation_relu(X, self._weights)
        else:
            raise ValueError("Unknown activation hyperparameter")

        self._predictions = np.argmax(predict_proba, 1)
        return self._predictions, predict_proba

    def score(self, X, y):
        try:
            predictions = self._predictions
        except Exception:
            predictions, _ = self.predict(X)
        labels = y

        return accuracy_score(predictions, labels)

