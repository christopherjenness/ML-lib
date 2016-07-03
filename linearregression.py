"""Least squares linear regression"""
import numpy as np
from gradientdescent import gradientdescent

class LinearRegression:
    """Class for implimenting Linear Regression"""
    def __init__(self):
        """
        Attributes::
            learned (bool): Keeps track of if Linear Regression has been fit
            weights (np.ndarray): vector of weights for linear regression
        """
        self.learned = False
        self.weights = np.NaN

    def predict(self, X):
        """
        Args:
            X (np.ndarray): Test data of shape[n_samples, n_features]

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
        prediction = np.dot(X, np.transpose(self.weights))
        return prediction

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
        hypothesis = np.dot(X, weights) - y
        gradient = np.dot(np.transpose(X), hypothesis)  / np.size(y)
        return gradient

    def fit(self, X, y, gradient=False):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]
            gradient (bool): Optional use of gradient descent to calculate weights
                if False, uses closed form solution to calculate weights.
            learned (bool): Keeps track of if model has been fit

        Returns:
            self: Returns an instance of self
        """
        y = np.asarray(y)
        X = np.asarray(X)
        X = np.column_stack((np.ones(np.shape(X)[0]), X))
        if gradient:
            self.weights = gradientdescent(X, y, self.grad)
        else:
            #Calculate weights (closed form solution)
            self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
        self.learned = True
        return self
