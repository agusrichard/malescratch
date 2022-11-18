import numpy as np


class GradientDescentRegressor:
    """Imitation of LinearRegression estimator from sklearn.
    It doesn't take any parameters or hyperparameters. 
    This estimator will use full-batch, which means that it uses
    all the data points and makes it slow to process a lot of data 
    points.

    Parameters: 
    ----------

    Attributes:
    ----------

    weights_ : array, shape (1, n_features) if n_classes == 2 else (n_classes,\
    n_features)
    Weights assigned to the features.

    bias_ : array, shape (1,) if n_classes == 2 else (n_classes,)
    Constants in decision function.

    """

    def __init__(self, learning_rate=0.1, step=10):
        self.weights_ = 0
        self.bias_ = 0
        self.learning_rate = learning_rate
        self.step = step

    def train(self, X, y):
        self.weights_ = np.random.randn(X.shape[1])
        self.bias_ = np.random.randn(1)

        for step in range(self.step):
            self.weights_ -= self.learning_rate * self._calculate_gradient(X, y)
            self.bias_ -= self.learning_rate * self._calculate_bias()

        return self

    def predict(self, X):
        y_pred = np.matmul(X, self.weights_) + self.bias_
        return y_pred

    def _calculate_gradient(self, X, y):
        gradient = np.empty(X.shape[1])
        for i in range(len(gradient)):
            summa = 0
            for j in range(X.shape[0]):
                summa = summa + (np.matmul(X[j], self.weights_) - y[j]) * X[j, i]
            summa = summa / X.shape[0]
            gradient[i] = summa

        return gradient

    def _calculate_bias(self):
        gradient = 0
        for j in range(X.shape[0]):
            gradient = gradient + (np.matmul(X[j], self.weights_) - y[j])

        return gradient / X.shape[0]

    def get_weights(self):
        return self.weights_

    def get_bias(self):
        return self.bias_
