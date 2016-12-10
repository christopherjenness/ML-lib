"""
Gaussian Mixture Model
"""
import random
import numpy as np
from scipy import stats

class KGaussian Mixture(object):
    """
    Gaussian Mixture classification
    """
    def __init__(self, k=5):
        """
        Attributes:
            samples (np.ndarray): Data of known target values
            values (np.ndarray): Known target values for data
            learned (bool): Keeps track of if model has been fit
            k (int): number of Guasian components
            gaussians (dict): dictionary of gausians with {groupID: [mean, variance]}
        """
        self.samples = np.nan
        self.values = np.nan
        self.k = k
        self.gaussians = {}
        self.learned = False

    def fit(self, X, y):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]

        Returns: an instance of self
        """
        self.samples = X
        self.values = y
        self._EM()
        self.learned = True
        return self
        
    def _EM(self):
        """
        Expectation-maximization algorithm for fitting gaussian mixtures
        """

    def predict(self, x):
        """
        Note: currenly only works on single vector and not matrices

        Args:
            x (np.ndarray): Training data of shape[1, n_features]

        Returns:
            float: Returns predicted class

        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
