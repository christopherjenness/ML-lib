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
    
def cross_entropy_error(y, predictions):
    """Cross Entropy Error"""
    n_samples = len(y)
    predictions[predictions==0] = 0.00000001
    predictions[predictions==1] = 0.99999999
    cross_entropy_errors = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
    return np.sum(cross_entropy_errors)
