import numpy as np

def GradientDescent(X, y, Gradient, alpha = 0.1, iterations = 1000, initial_weights = False, stochastic = False):
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
    """
    # If no initial weights given, initials weights = 0
    if initial_weights == False:
        weights= np.zeros(np.shape(X)[1])
    else:
        weights = initial_weights
    # Update weights via gradient descent
    iteration = 0
    while iteration < iterations:
        weights = weights - alpha * Gradient(X, y, weights)
        iteration +=1
    return weights
        
    
    
    
              