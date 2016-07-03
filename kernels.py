"""
Kernels which can be used by SVMs.
"""

import numpy as np
import math

def linear(x1, x2, *args):
    return np.dot(x1, x2.T)

def polynomial(x1, x2, power, coef, *args):
    return (np.dot(x1, x2) + c)**power

def rbf(x1, x2, gamma):
    difference = x1 - x2
    return np.exp(-gamma * (np.dot(difference, difference.T)**2))

def tanh(x1, x2, k, c):
    return math.tanh(k * np.dot(x1, x2) + c)