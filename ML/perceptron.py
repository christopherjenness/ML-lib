"""Perceptron classifier of binary data"""
import numpy as np

class Perceptron:
    """
    Perceptron classifier of binary data.  If data is not linearly seperable data,
    the pocket algorithm is highly recommended
    """
    def __init__(self, max_iter=100, learning_rate=1, pocket=False, initial_weights=False):
        """
        Args:
            max_iter (int): Maximum number of iterations through PLA before stoping
            learning_rate (float): PLA step size
            pocket (bool): optional incorporation of pocket algorithm,
                saves best weights incase model is not fit
            initial_weights (np.ndarray): optional initial weights for PLA

        Attributes::
            weights (np.ndarray): vector of weights for linear separation
            learned (bool): Keeps track of if perceptron has been fit
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.pocket = pocket

        self.weights = initial_weights
        self.learned = False

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
        X = np.column_stack((np.ones(np.shape(X)[0]), X))
        prediction = np.sign(np.dot(X, np.transpose(self.weights)))
        return prediction

    def fit(self, X, y):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]

        Returns:
            self: Returns an instance of self

        Raises:
            ValueError if y contains values other than -1 and 1
        """
        y = np.asarray(y)
        X = np.asarray(X)
        X = np.column_stack((np.ones(np.shape(X)[0]), X))
        #Check if y contains only 1 or -1 values
        if False in np.in1d(y, [-1, 1]):
            raise NameError('y required to contain only 1 and -1')
        if isinstance(self.weights, bool):
            self.weights = np.zeros(np.shape(X)[1])
        pocket_weights = np.zeros(np.shape(X)[1])
        # Update weights until they linearly separate inputs
        # if self.pocket, keep track of best weights
        iteration = 0
        while not np.array_equal(np.sign(np.dot(X, np.transpose(self.weights))), y):
            if iteration > self.max_iter:
                if self.pocket:
                    self.weights = pocket_weights
                else:
                    self.weights = np.NaN
                return self
            classification = np.equal(np.sign(np.dot(X, np.transpose(self.weights))), y)
            #find misclassified point, and update classifier to correct misclassified point
            misclassified = np.where(classification == False)[0][0]
            updated_weights = self.weights + self.learning_rate * y[misclassified] * X[misclassified]
            if self.pocket:
                if np.sum(classification) > np.sum(np.equal(np.sign(np.dot(X, np.transpose(pocket_weights))), y)):
                    pocket_weights = updated_weights
            self.weights = self.weights + self.learning_rate * y[misclassified] * X[misclassified]
            iteration += 1
        self.learned = True
        return self
