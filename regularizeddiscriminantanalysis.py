"""
Regularized discriminant analysis is a compromise between 
linear discrimenent analysis and quadratic discrimenent analysis.
"""

import numpy as np

class RegularizedDisriminentAnalysis():

    def __init__(self, alpha):
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
        self.weights = np.NaN
        self.alpha = alpha
        
    def fit(self, X, y):
        n_samples = np.shape(X)[0]
        n_samples = np.shape(X)[1]
        n_classes = len(np.unique(y))
        
        class_covariances = {}
        class_priors = {}
        class_means = {}
        pooled_covariances = 0
        
        for i in range(n_classes):
            class_indices = np.where(y == i)
            class_samples = X[class_samples, :]
            
            class_priors[i] = float(len(class_indices)) / len(y)
            class_means[i] = np.mean(class_samples, axis = 0)
            
            class_covariances[i] = np.cov(class_samples, rowvar=0)
        
        print (class_covariances)
