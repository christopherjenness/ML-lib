import numpy as np

class LinearRegression:
    def __init__(self):
        """
        Attributes::
            learned (bool): Keeps track of if perceptron has been fit
            weights (np.ndarray): vector of weights for linear separation
        """
        self.learned = False
        self.weights = np.NaN
        

    def predict(self, X):
        """
        Args: 
            X (np.ndarray): Training data of shape[n_samples, n_features]
                
        Returns:
            prediction (np.ndarray): shape[n_samples 1)
                Returns predicted values       
                
        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
        # Add column of 1s to X for perceptron threshold
        X = np.asarray(X)
        if X.ndim==1:
            X = np.insert(X, 0, 1, axis = 1)
        else:
            X = np.insert(X, 0, 1, axis = 1)
        prediction = np.dot(X, np.transpose(self.weights))
        return prediction
    
    def fit(self, X, y):
        """
        Args: 
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1] 
                (vals must be -1 or 1)
                
        Returns:
            self: Returns an instance of self
        Raises:
            ValueError if y contains values other than -1 or 1
        """
        y = np.asarray(y)
        X = np.asarray(X)
        X = np.insert(X, 0, 1, axis = 1)
        #Calculate weights (closed form solution)
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
        self.learned = True
        return self