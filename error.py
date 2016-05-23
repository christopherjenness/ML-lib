"""Common error measures for model predictions when true answer is known"""

import numpy as np

def mse(y, predictions):
    """Mean Square Error"""
    n_samples = len(y)
    differences = y - predictions
    return np.dot(differences, differences.T) / n_samples

def mean_classification_error(y, predictions):
    """Average classiication error"""
    n_samples = len(y)
    correct_classifications = np.equal(y, predictions)
    return (n_samples - correct_classifications.sum()) / n_samples

