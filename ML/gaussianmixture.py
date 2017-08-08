"""
Gaussian Mixture Model
"""

import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixture(object):
    """
    Gaussian Mixture classification
    """
    def __init__(self, c=2):
        """
        Attributes:
            samples (np.ndarray): Data of known target values
            values (np.ndarray): Known target values for data
            learned (bool): Keeps track of if model has been fit
            c (int): number of Guasian components
            gaussians (dict): dictionary of gausians with
                {groupID: [mean, variance]}
        """
        self.samples = np.nan
        self.values = np.nan
        self.mus = np.nan
        self.covs = np.nan
        self.priors = np.nan
        self.responsibility_matrix = np.nan
        self.c = c
        self.gaussians = {}
        self.learned = False

    def fit(self, X, iterations=50):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]

        Returns: an instance of self
        """
        self.samples = X
        n_samples = np.shape(X)[0]
        n_features = np.shape(X)[1]

        # Initialize mus, covs, priors
        initial_indices = np.random.choice(range(n_samples), self.c,
                                           replace=False)

        self.mus = X[initial_indices, :]
        self.covs = [np.identity(n_features) for i in range(self.c)]
        self.priors = [1.0/self.c for i in range(self.c)]
        for iteration in range(iterations):
            self.responsibility_matrix = self._expectation()
            self.priors, self.mus, self.covs = self._maximization()
        self.learned = True
        return self

    def _expectations(self, point):
        responsibilities = [0 for i in range(self.c)]
        for k in range(self.c):
            probability = multivariate_normal.pdf(point,
                                                  mean=self.mus[k],
                                                  cov=self.covs[k]) * \
                                                      self.priors[k]
            responsibilities[k] = probability
        responsibilities = [float(i) / sum(responsibilities)
                            for i in responsibilities]
        return responsibilities

    def _expectation(self):
        return np.apply_along_axis(self._expectations, 1, self.samples)

    def _maximization(self):
        # Maximize priors
        priors = sum(self.responsibility_matrix)
        priors = [float(i)/sum(priors) for i in priors]

        # Maximize means
        mus = [0 for i in range(self.c)]
        for k in range(self.c):
            mus_k = sum(np.multiply(self.samples,
                                    self.responsibility_matrix[:, k][:, np.newaxis]))
            normalized_mus_k = mus_k / sum(self.responsibility_matrix[:, k])
            mus[k] = normalized_mus_k

        # Maximize covariances
        covs = [0 for i in range(self.c)]
        for k in range(self.c):
            covs[k] = np.cov(self.samples.T,
                             aweights=self.responsibility_matrix[:, k])

        return priors, mus, covs

    def predict(self, x, probs=False):
        """
        Note: currenly only works on single vector and not matrices

        Args:
            x (np.ndarray): Training data of shape[1, n_features]
            probs (bool): if True, returns probability of each class as well

        Returns:
            float: Returns predicted class

        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')

        probabilities = [0 for i in range(self.c)]
        for k in range(self.c):
            probability = multivariate_normal.pdf(x,
                                                  mean=self.mus[k],
                                                  cov=self.covs[k]) * \
                                                  self.priors[k]
            probabilities[k] = probability
        max_class = np.argmax(probabilities)
        class_probs = [float(i)/sum(probabilities) for i in probabilities]
        if probs:
            return (max_class, class_probs)
        return np.argmax(probabilities)
