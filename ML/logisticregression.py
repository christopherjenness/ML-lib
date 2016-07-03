"""Logistic Regression classifier"""
import numpy as np
from gradientdescent import gradientdescent

class LogisticRegression:
    """Logistic Regression classifier with gradient descent implementation"""
    def __init__(self):
        """
        Attributes:
            learned (bool): Keeps track of if model has been fit
            weights (np.ndarray): vector of weights for linear separation
        """
        self.learned = False
        self.weights = np.NaN

    @staticmethod
    def logistic_function(logistic_input):
        """
        Args:
            logistic_input (np.ndarray): array of shape[n_samples, 1]

        Returns:
            logistic: shape[n_samples, 1]
                Returns logistic transformation of data
        """
        return 1 / (1 + np.exp(-logistic_input))

    def grad(self, X, y, weights):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]
            weights (np.ndarray): Optional use of gradient descent to calculate weights
                if False, uses closed form solution to calculate weights.

        Returns:
            gradient: Returns the gradient of the linear regression cost function
        """
        hypothesis = self.logistic_function(np.dot(X, weights)) - y
        gradient = np.dot(np.transpose(X), hypothesis)  / np.size(y)
        return gradient

    def predict(self, X):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]

        Returns:
            prediction (np.ndarray): shape[n_samples, 1]
                Returns predicted values

        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
        # Add column of 1s to X for perceptron threshold
        X = np.asarray(X)
        X = np.column_stack((np.ones(np.shape(X)[0]), X))
        prediction = self.logistic_function(np.dot(X, np.transpose(self.weights)))
        return np.round(prediction)

    def fit(self, X, y, reg_parameter=0):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]
            reg_parameter (float): float to determine strength of regulatrization  penalty
                if 0, then no linear regression without regularization is performed

        Returns:
            self: Returns an instance of self

        Raises:
            ValueError if y contains values other than 0 and 1
        """
        y = np.asarray(y)
        if False in np.in1d(y, [0, 1]):
            raise NameError('y required to contain only 0 and 1')
        X = np.asarray(X)
        X = np.column_stack((np.ones(np.shape(X)[0]), X))
        self.weights = gradientdescent(X, y, self.grad, reg_param=reg_parameter)
        self.learned = True
        return self
