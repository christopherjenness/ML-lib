"""
This module includes descent methods for finding local minima.
"""
import numpy as np

def gradientdescent(X, y, gradient, cost=None, alpha=0.01, iterations=10000,
                    initial_weights=False, stochastic=False, reg_param=0,
                    backtrack_line_search=False, backtrack_alpha=0.5,
                    backtrack_beta=0.5):
    """
    Args:
        X (np.ndarray): Training data of shape[n_samples, n_features]
        y (np.ndarray): Target values of shape[n_samples, 1]
        gradient (function): Function to compute the gradient
            Gradient is a function of (X, y, weights)
        cost (function): Cost function (required if using exact line search
            or backtrack line search).
        alpha (float): step size during each gradient descent iteration
        iterations (int): Number of iterations of gradient descent to perform
        initial_weights (np.ndarray): initial weights for gradient descent
        stochastic (bool): If True, implement Stochastic Gradient Descent
        reg_param (float): value of regularization parameter.
            if 0, function performs gradient descent without regularization
        backtrack_line_search (bool): If True, perform backtracking line search
            to determine step size.
        backtrack_alpha (float): In range(0, 0.5), accepted decrease in function,
            based on extrapolation
        backtrack_beta (float): In range(0, 1), how quickly step size is updated
            when calculating backtrack.

    Returns:
        (np.ndarray): shape[n_features, 1], array of weights

    Notes:
        Currently, backtracking line search is incompatable with
        stochastic=True
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
        if backtrack_line_search:
            step_size = 1
            while True:
                if step_size < 0.0001:
                    break
                step_size *= backtrack_beta
                estimate_weights = weights - step_size * gradient(X, y, weights)
                estimate_cost = cost(X, y, estimate_weights)
                bound = cost(X, y, weights)  - backtrack_alpha * step_size * (np.linalg.norm(gradient(X, y, weights)))**2
                if estimate_cost < bound:
                    weights = estimate_weights
                    break
        else:
            weights = weights * (1 - alpha * reg_param / len(y)) - alpha * gradient(X, y, weights)
        iteration += 1
    return weights

def steepestdescent(X, y, gradient, alpha=0.01, iterations=10000,
                    initial_weights=False, norm="L1"):
    """
    Args:
        X (np.ndarray): Training data of shape[n_samples, n_features]
        y (np.ndarray): Target values of shape[n_samples, 1]
        gradient (function): Function to compute the gradient
            Gradient is a function of (X, y, weights)
        alpha (float): step size during each gradient descent iteration
        iterations (int): Number of iterations of gradient descent to perform
        initial_weights (np.ndarray): initial weights for gradient descent
        norm {'L1'}: Norm used to find steepest descent direction
            L1: Coordinant gradient descent. Each step is in the coordinant
                direciton that is steepest.

    Returns:
        np.ndarray: shape[n_features, 1] array of weights

    Notes:
        Currently, only L1 norm is implimented.  In the future, other norms
        can be added.
    """
    # If no initial weights given, initials weights = 0
    if not initial_weights:
        weights = np.zeros(np.shape(X)[1])
    iteration = 0
    while iteration < iterations:
        if norm == "L1":
            gradients = gradient(X, y, weights)
            steepest_direction = np.absolute(gradients).argmax()
            steepest_descent = np.zeros(len(weights))
            steepest_descent[steepest_direction] = gradients[steepest_direction]
            weights = weights - alpha * steepest_descent
        iteration += 1
    return weights

def newtonsmethod(X, y, gradient, hessian, alpha=0.01, iterations=10000,
                  initial_weights=False):
    """
    Args:
        X (np.ndarray): Training data of shape[n_samples, n_features]
        y (np.ndarray): Target values of shape[n_samples, 1]
        gradient (function): Function to compute the gradient
            Gradient is a function of (X, y, weights)
        hessien (function): Function to compute the Hessian
            Hessian is a function of (X)
        alpha (float): step size during each gradient descent iteration
        iterations (int): Number of iterations of gradient descent to perform
        initial_weights (np.ndarray): initial weights for gradient descent

    Returns:
        np.ndarray: shape[n_features, 1] array of weights

    Notes:
        Pure Newton method (constant step size) is implemented.  Damped Newton
        method will be implimented in the future using backtracking algorithm (see
        gradient descent method for backtracking implimentation).
    """
    # If no initial weights given, initials weights = 0
    if not initial_weights:
        weights = np.zeros(np.shape(X)[1])
    iteration = 0
    while iteration < iterations:
        step_direction = -np.dot(np.linalg.pinv(hessian(X)), gradient(X, y, weights))
        weights = weights + alpha * step_direction
        iteration += 1
    return weights

