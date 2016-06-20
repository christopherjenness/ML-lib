"""
Kernel Methods: 
"""

import numpy as np

class KernelMethods(object):
    
    def __init__(self):
        self.learned = False
        self.X = np.NaN
        self.y = np.NaN
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.learned = True

    @staticmethod    
    def epanechnikovkernel(x0, x, gamma):
        t = np.linalg.norm(x - x0) / gamma
        if t > 1:
            return 0
        return 0.75 * (1 - t**2)
    
    @staticmethod    
    def tricubekernel(x0, x, gamma):
        t = np.linalg.norm(x - x0) / gamma
        if t > 1:
            return 0
        return (1 - t**3)**3
        
    def nadarayaaverage(self, x, kernel, gamma):
        if self.learned == False:
            raise NameError('Please fit model first')
        numerator = 0
        denominator = 0
        for row in range(np.shape(self.X)[0]):
            numerator += kernel(self.X[row, :], x, gamma) * self.y[row]
            denominator += kernel(self.X[row, :], x, gamma)
            print(numerator, denominator)
        return numerator / denominator
        
    def locallinearregression():
        """
        Local linear regression eliminates bias at boundries
        of domain.
        """
        return True
        
    def localpolynomialregression():
        """
        Local polynomial regression eliminates bias at internal
        curvature of domain.
        """
        return True












