import numpy as np

def MSE(y, predictions):
    n_samples = len(y)
    differences = y - predictions
    return np.dot(differences, differences.T) / n_samples