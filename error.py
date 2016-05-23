import numpy as np

def MSE(y, predictions):
    n_samples = len(y)
    differences = y - predictions
    return np.dot(differences, differences.T) / n_samples

def mean_classification_error(y, predictions):
    n_samples = len(y)
    correct_classifications = np.equal(y, predictions)
    return (n_samples - correct_classifications.sum()) / n_samples

