"""
Linear, Quadratic and Regularized discriminant analysis.
Regularized discriminant analysis is a compromise between
linear discrimenent analysis and quadratic discrimenent analysis.
"""

import numpy as np

class DisriminentAnalysis():
    """
    Class for implimenting Regularized Discriminent Analysis
    LDA is performed when alpha == 1
    QDA is performed when alpha == 2
    """
    def __init__(self, alpha=1):
        """
        Attributes::
            learned (bool): Keeps track of if RDA has been fit
            alpha (float): regularization parameter in range [0, 1]
                When alpha == 1, LDA is performed, when alpha == 2, QDA is performed.
                Intermediate values of alpha tradeoff pooled covarainces of LDA
                with seperate covariances of QDA.
            class_names (np.ndarray): array of class names. [0, 1] for example.
            class_priors (dict): prior probability of each class.
                determined via fraction of training samples in each class
            class_means (dict): vector means of each class
            regularized_covariances (np.ndarray): RDA covariances.
                weighted combination of QDA class covariance and LDA pooled covariance
        """
        self.learned = False
        self.alpha = alpha
        self.class_names = []
        self.class_priors = {}
        self.class_means = {}
        self.regularized_covariances = {}

    def fit(self, X, y):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]

        Returns:
            self: Returns an instance of self
        """
        self.class_names = np.unique(y)
        class_covariances = {}
        pooled_covariances = 0
        #calculate class priors, class means, and LDA pooled covariance matrix
        for i in self.class_names:
            class_indices = np.where(y == i)[0]
            class_samples = X[class_indices, :]
            self.class_priors[i] = float(len(class_indices)) / len(y)
            self.class_means[i] = np.mean(class_samples, axis=0)
            class_covariances[i] = np.cov(class_samples, rowvar=0)
            #add contribution of individual class covariance to the pooled covariance)
            pooled_covariances += class_covariances[i] * self.class_priors[i]
        #calculate RDA regularized covariance matricies for each class
        for i in self.class_names:
            self.regularized_covariances[i] = self.alpha * class_covariances[i] + (1 - self.alpha) * pooled_covariances
        self.learned = True
        return self

    def predict(self, x):
        """
        Args:
            x (np.array): Training data of shape[1, n_features]
                Currently, only vector of single sample is supported
        Returns:
            prediction: Returns predicted class of sample

        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
        #Determine probability of each class given input vector
        class_deltas = {}
        for i in self.class_names:
            # Divid the class delta calculation into 3 parts
            part1 = -0.5 * np.linalg.det(self.regularized_covariances[i])
            part2 = -0.5 * np.dot(np.dot((x - self.class_means[i]).T, np.linalg.pinv(self.regularized_covariances[i])), (x - self.class_means[i]))
            part3 = np.log(self.class_priors[i])
            class_deltas[i] = part1 + part2 + part3
        return max(class_deltas, key=class_deltas.get)
