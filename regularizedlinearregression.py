"""
Least squares linear regression with weight decay regularization.
In the future, this should be merged with LinearRegression parent class.
"""

import numpy as np
from linearregression import LinearRegression
from gradientdescent import gradientdescent

class RegularizedLinearRegression(LinearRegression):
    """Linear regression implementation with regularization"""
    def fit(self, X, y, gradient=False, reg_parameter=0.01):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]
            gradient_descent (bool): Optional use of gradient descent to calculate weights
                if False, uses closed form solution to calculate weights.
            reg_parameter (float): float to determine strength of regulatrization  penalty
                if 0, then no linear regression without regularization is performed

        Returns:
            self: Returns an instance of self
        """
        y = np.asarray(y)
        X = np.asarray(X)
        X = np.column_stack((np.ones(np.shape(X)[0]), X))
        if gradient:
            self.weights = gradientdescent(X, y, self.grad, reg_param=reg_parameter)
        else:
            #Calculate weights (closed form solution)
            XtX_lambaI = np.dot(np.transpose(X), X) + reg_parameter * np.identity(len(np.dot(np.transpose(X), X)))
            self.weights = np.dot(np.linalg.pinv(XtX_lambaI), np.dot(np.transpose(X), y))
        self.learned = True
        return self
