"""
Regularized discriminant analysis is a compromise between 
linear discrimenent analysis and quadratic discrimenent analysis.
"""

import numpy as np
import itertools

class RegularizedDisriminentAnalysis():

    def __init__(self, alpha = 1):
        """
        Attributes::
            learned (bool): Keeps track of if Linear Regression has been fit
            weights (np.ndarray): vector of weights for linear regression
            alpha (float): regularization parameter in range [0, 1]
                When alpha == 1, LDA is performed, when alpha == 2, QDA is performed.
                Intermediate values of alpha tradeoff pooled covarainces of LDA
                with seperate covariances of QDA.
        """
        self.learned = False
        self.alpha = alpha
        self.class_names = []
        self.regularized_covariances = {}
        self.class_priors = {}
        self.class_means = {}
        self.class_names = []
        
    def fit(self, X, y):
        n_samples = np.shape(X)[0]
        n_samples = np.shape(X)[1]
        n_classes = len(np.unique(y))
        
        self.class_names = np.unique(y)
        class_covariances = {}
        pooled_covariances = 0
        
        for i in self.class_names:
            class_indices = np.where(y == i)[0]
            class_samples = X[class_indices, :]
            
            self.class_priors[i] = float(len(class_indices)) / len(y)
            self.class_means[i] = np.mean(class_samples, axis = 0)
            
            class_covariances[i] = np.cov(class_samples, rowvar=0)
            #add contribution of individual class covariance to the pooled covariance)
            pooled_covariances += class_covariances[i] * self.class_priors[i]
        for i in self.class_names:
            self.regularized_covariances[i] = self.alpha * class_covariances[i] + (1 - self.alpha) * pooled_covariances
        self.learned = True
        
    def predict(self, x):
        class_deltas = {}
        for i in self.class_names:
            # Divid the class delta calculation into 3 parts
            part1 = -0.5 * np.linalg.det(self.regularized_covariances[i])
            part2 = -0.5 * np.dot(np.dot((x - self.class_means[i]).T,  np.linalg.pinv(self.regularized_covariances[i])), (x - self.class_means[i]))
            part3 = np.log(self.class_priors[i])
            class_deltas[i] = part1 + part2 + part3
        return max(class_deltas, key = class_deltas.get)
        
