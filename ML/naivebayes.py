"""
Naive Bayes Classifier
Includes gaussian, bernoulli and multinomial models
"""

import abc
import numpy as np

class NaiveBayes:
    """
    Naive Bayes Classifier
    Given class label, assumes features are independent
    """
     __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        Attributes:
            learned (bool): Keeps track of if classifier has been fit
            class_names (np.ndarray): array of class names. [0, 1] for example.
            class_priors (dict): prior probability of each class.
                determined via fraction of training samples in each class
            class_parameters (dict): dict of parameters for each class

        """
        self.learned = False
        self.class_names = []
        self.class_priors = {}
        self.class_parameters = {}
        
    @abc.abstractmethod
    def fit(self, X, y):
        """
        Fits Naive Bayes classifier
        
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]

        Returns: an instance of self
        
        """
        return self
        
    @abc.abstractmethod
    def predict(self, X):
        """
        Args:
            x (np.array): Training data of shape[1, n_features]
                Currently, only vector of single sample is supported
                
        Returns: predicted class of sample

        Raises:
            ValueError if model has not been fit
        """
        return self

