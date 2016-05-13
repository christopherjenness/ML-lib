"""
This module includes the gradient descent algorithm.  By taking small steps
down the negative of the gradient, local minima are reached.
"""
import numpy as np

def gradientdescent(X, y, gradient, alpha=0.01, iterations=10000,
                    initial_weights=False, stochastic=False, reg_param=0):
    """
    Args:
        X (np.ndarray): Training data of shape[n_samples, n_features]
        y (np.ndarray): Target values of shape[n_samples, 1]
        Gradient (function): Function to compute the gradient
            Gradient is a function of (X, y, weights)
        alpha (float): step size during each gradient descent iteration
        iterations (int): Number of iterations of gradient descent to perform
        initial_weights (np.ndarray): initial weights for gradient descent
        stochastic (bool): If True, implement Stochastic Gradient Descent
        reg_param (float): value of regularization parameter.
            if 0, function performs gradient descent without regularization
    Returns:
        weights (np.ndarray): shape[n_features, 1]
                Returns array of weights
    """
    # If no initial weights given, initials weights = 0
    if not initial_weights:
        weights = np.zeros(np.shape(X)[1])
    else:
        weights = initial_weights
    # Update weights via gradient descent
    iteration = 0
    while iteration < iterations:
        if stochastic:
            random_index = np.random.randint(len(y))
            weights = weights - alpha * gradient(X[random_index], y[random_index], weights)

        else:
            weights = weights * (1 - alpha * reg_param / len(y)) - alpha * gradient(X, y, weights)
        iteration += 1
    return weights
