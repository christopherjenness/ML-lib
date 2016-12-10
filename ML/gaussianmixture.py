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
            c (int): number of Guasian components
            gaussians (dict): dictionary of gausians with {groupID: [mean, variance]}
        """
        self.samples = np.nan
        self.values = np.nan
        self.c = c
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
        
    def _expectations(point):
        responsibilities = [0 for i in range(self.c)]
        for k in range(self.c):
            probability = multivariate_normal.pdf(point, mean = self.mus[k], cov = self.covs[k]) * self.priors[k]
            responsibilities[k] = probability
        responsibilities = [float(i)/sum(responsibilities) for i in responsibilities]
        return responsibilities
        
    def _expectation():
        return np.apply_along_axis(self.expectations, 1, self.X)
        
    def maximization():
        # Maximize priors
        priors = sum(self.responsibility_matrix)
        priors = [float(i)/sum(priors) for i in priors]
        
        # Maximize means
        mus = [0 for i in range(self.c)]
        for k in range(self.c):
            mus_k = sum(np.multiply(self.X, self.responsibility_matrix[:, k][:,np.newaxis]))
            normalized_mus_k = mus_k / sum(self.responsibility_matrix[:,k])
            mus[k] = normalized_mus_k
        
        # Maximize covariances
        covs = [0 for i in range(self.c)]
        for k in range(self.c):
            covs[k] = np.cov(self.X.T, aweights=self.responsibility_matrix[:,k])
        
        return priors, mus, covs
        

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
