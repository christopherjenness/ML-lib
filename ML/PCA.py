"""
Exact principal component analysis (PCA)
"""

class PCA(object):
    """
    Exact principal component analysis (PCA)
    """

    def __init__(self):
        self.X = None
        self.X_normalized = None
        self.transformed_X = None

    def fit(self, X):
        self.X = X
        self.mean_normalize()
        cov_matrix = (np.dot(self.X_normalized.T, self.X_normalized)) * 1/(np.shape(self.X)[0] - 1)
        eigen_vals, eigen_vectors = np.linalg.eig(cov_matrix)
        eigen_vectors = eigen_vectors.T
        # Sort the eigen vectors
        eigen_order = np.argsort(eigen_vals)
        P = eigen_vectors[:,eigen_order]
        print(cov_matrix, P)
        # Project X onto PC
        self.transformed_X = (np.dot(P, X.T)).T
        return self

    def mean_normalize(self):
        col_means = self.X.sum(axis=0) / np.shape(self.X)[0]
        self.X_normalized = self.X - col_means
