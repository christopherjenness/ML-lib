"""
Exact principal component analysis (PCA)
"""
import numpy as np


class PCA(object):
    """
    Exact principal component analysis (PCA).  Transforms given data set
    into orthonormal basis, maximizing variance.
    TODO return variance explained by each principal component
    """

    def __init__(self):
        """
        Attributes:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            X_normalized (np.ndarray): Mean normalized data of
                shape [n_samples, n_features]
            transformed_X (np.ndarray): Transformed data of
                shape [n_samples, n_features]
                This data has contains all principal components,
                    in order of variance explained
        """
        self.X = None
        self.X_normalized = None
        self.transformed_X = None

    def fit(self, X):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
        Returns: an instance of self
        """
        self.X = X
        self.mean_normalize()
        cov_matrix = (np.dot(self.X_normalized.T, self.X_normalized)) * \
                      1 / (np.shape(self.X)[0] - 1)
        eigen_vals, eigen_vectors = np.linalg.eig(cov_matrix)
        eigen_vectors = eigen_vectors.T
        # Sort the eigen vectors
        eigen_order = np.argsort(eigen_vals)
        P = eigen_vectors[:, eigen_order]
        # Project X onto PC
        self.transformed_X = (np.dot(P, X.T)).T
        return self

    def mean_normalize(self):
        """
        Normalizes input data (subtracts mean from each sample)
        """
        col_means = self.X.sum(axis=0) / np.shape(self.X)[0]
        self.X_normalized = self.X - col_means
