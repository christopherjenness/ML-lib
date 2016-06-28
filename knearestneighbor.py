"""
K nearest neighbors
"""
import numpy as np
from scipy import stats

class KNearestNeighbor(object):
    """
    K nearest neighbors classification and regression
    """
    def __init__(self):
        """
        Attributes:
            samples (np.ndarray): Data of known target values
            values (np.ndarray): Known target values for data
            learned (bool): Keeps track of if model has been fit
        """
        self.samples = np.nan
        self.values = np.nan
        self.learned = False

    def fit(self, X, y):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]

        Returns:
            self: Returns an instance of self
        """
        self.samples = X
        self.values = y
        self.learned = True
        return self

    def predict(self, x, k=1, model='regression'):
        """
        Note: currenly only works on single vector and not matrices

        Args:
            x (np.ndarray): Training data of shape[1, n_features]
            k (int): number of nearest neighbor to consider
            model: {'regression', 'classification'}
                K nearest neighbor classification or regression.
                Choice most likely depends on the type of data the model was fit with.

        Returns:
            prediction (float): Returns predicted value

        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
        distances = np.array([])
        for row in range(np.shape(self.samples)[0]):
            #add distance from x to sample row to distances vector
            distances = np.append(distances, np.linalg.norm(x - self.samples[row, :]))
            nearestneighbors = distances.argsort()[:k]
        if model == 'regression':
            prediction = self.values[nearestneighbors].mean()
        if model == 'classification':
            prediction = stats.mode(self.values[nearestneighbors]).mode
        return prediction
