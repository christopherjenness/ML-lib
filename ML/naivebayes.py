"""
Naive Bayes Classifier
Includes gaussian, bernoulli and multinomial models
"""

import abc
import numpy as np
from scipy.stats import norm

class NaiveBayes():
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
        
class GaussianNaiveBayes(NaiveBayes):
    
    def __init__(self):
        super().__init__()
        
    def fit(self, X, y):
        n_samples = len(y)
        self.class_names = list(np.unique(y))
        for class_name in self.class_names:
            # Compute class priors
            class_prior = float(len(y[y == class_name]) / n_samples)
            self.class_priors[class_name] = class_prior
            
            # Compute class mean and variance
            # Assume features are independent given class label
            mu = list(np.mean(X[y==class_name,:], axis=0))
            variance = list(np.var(X[y==class_name,:], axis=0))
            self.class_parameters[class_name] = [mu, variance]
        return self
        
    def predict(self, x, probabilities=False):
        # Calculate non-normalized class probabilities and find
        # highest class probability
        classification_class = None
        classification_probability = -np.inf
        class_probabilities = {}
        normalizing_constant = 0
        for class_name in self.class_names:
            class_mu = self.class_parameters[class_name][0]
            class_variance = self.class_parameters[class_name][1]
            numerator = self.class_priors[class_name]
            for feature in range(len(x)):
                numerator *= norm.pdf(x[feature], 
                                                     loc=class_mu[feature], 
                                                     scale=class_variance[feature])
            normalizing_constant += numerator
            class_probabilities[class_name] = numerator
            if numerator > classification_probability:
                classification_probability = numerator
                classification_class = class_name
        if not probabilities:
            return classification_class

        for class_name in self.class_names:
            class_probabilities[class_name] /= normalizing_constant
        
        return class_probabilities
 
### Begin Scratch Work       
y = np.array([0, 0, 1, 1, 0, 0, 2, 2])
x1 = np.array([np.random.normal() for i in range(8)])
x2 = np.array([np.random.normal() for i in range(8)])
x = np.array([0.1, -0.01])
X = np.array([x1, x2]).T

a = GaussianNaiveBayes()
a.fit(X, y) 
print (a.predict(x))



















