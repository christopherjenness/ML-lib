import numpy as np
from Linear_Regression import LinearRegression
from Gradient_Descent import GradientDescent


class RegularizedLinearRegression(LinearRegression):
    def fit(self, X, y, gradient_descent = False, reg_parameter = 0.01):
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
        if gradient_descent:
            # Use gradient descent to calculate weights
            def LRgrad(X, y, weights):
                hypothesis = np.dot(X, weights) - y
                return np.dot(np.transpose(X), hypothesis)  / np.size(y)
            self.weights  = GradientDescent(X, y, LRgrad, reg_param = reg_parameter)
        else:
            #Calculate weights (closed form solution)
            XtX_lambaI = np.dot(np.transpose(X), X) + reg_parameter * np.identity(len(np.dot(np.transpose(X), X)))
            self.weights = np.dot(np.linalg.pinv(XtX_lambaI), np.dot(np.transpose(X), y))
        self.learned = True
        return self    

    