import numpy as np
from Gradient_Descent import GradientDescent

class LinearRegression:
    def __init__(self):
        """
        Attributes::
            learned (bool): Keeps track of if perceptron has been fit
            weights (np.ndarray): vector of weights for linear separation
        """
        self.learned = False
        self.weights = np.NaN
    
    def logistic_function(self, logistic_input):
        """
        Args: 
            logistic_input (np.ndarray): array of shape[n_samples, 1]
            
        Returns:
            logistic: shape[n_samples, 1]
                Returns logistic transformation of data     
        """    
        return 1 / (1 + np.exp(logistic_input)

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
        X = np.column_stack((np.ones(len(a)), X))
        prediction = self.logistic(np.dot(X, np.transpose(self.weights)))
        return prediction
    
    def fit(self, X, y):
        """
        Args: 
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1] 
                
        Returns:
            self: Returns an instance of self
        """
        return self