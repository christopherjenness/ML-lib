import numpy as np
from Gradient_Descent import GradientDescent

class LinearRegression:
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
        prediction = np.dot(X, np.transpose(self.weights))
        return prediction
    
    def fit(self, X, y, gradient_descent = False):
        """
        Args: 
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1] 
            gradient_descent (bool): Optional use of gradient descent to calculate weights
                if False, uses closed form solution to calculate weights.
                
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
            self.weights  = GradientDescent(X, y, LRgrad)
        else:
            #Calculate weights (closed form solution)
            self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
        self.learned = True
        return self