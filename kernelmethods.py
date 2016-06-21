"""
Kernel Methods: 
"""

import numpy as np

class KernelMethods(object):
    
    def __init__(self):
        self.learned = False
        self.X = np.NaN
        self.X_intercept = np.NaN
        self.y = np.NaN
        
    def fit(self, X, y):
        self.X = np.asarray(X)
        self.X_intercept = np.column_stack((np.ones(np.shape(X)[0]), X))
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
        x = np.array(x)
        if not self.learned:
            raise NameError('Please fit model first')
        numerator = 0
        denominator = 0
        for row in range(np.shape(self.X)[0]):
            numerator += kernel(self.X[row, :], x, gamma) * self.y[row]
            denominator += kernel(self.X[row, :], x, gamma)
            print(numerator, denominator)
        return numerator / denominator
        
    def locallinearregression(self, x, kernel, gamma):
        """
        Local linear regression eliminates bias at boundries
        of domain.  It uses weighted least squares, determining
        weights from the kernel.
        """
        W = []
        for row in range(np.shape(self.X)[0]):
            W.append(kernel(self.X[row, :], x, gamma))
        W = np.diag(W)
        #Calculate solution using closed form solution
        solution_a = np.linalg.pinv(self.X_intercept.T.dot(W).dot(self.X_intercept))
        solution_b = self.X_intercept.T.dot(W).dot(self.y)
        solution = np.dot(solution_a, solution_b)
        x = np.asarray(x)
        x = np.column_stack((1, x))
        prediction = np.inner(x, solution)
        return float(prediction)

    @staticmethod
    def kerneldensityestimate(samples, x, gamma):
        samples = np.matrix(samples)
        N = len(x)
        estimate = 0
        for row in range(np.shape(samples)[0]):
            x_i = np.array(x) - samples[row, :]
            print(x_i)
            gaussiankernel = (1/(2*np.pi)**0.5 * gamma**2)**N * np.exp(-np.linalg.norm(x_i)**2 / (2 * gamma**2)) 
            print(gaussiankernel)
            estimate += gaussiankernel
        estimate /= N
        return estimate
        
    def kerneldensitypredict(self, x, gamma):
        if not self.learned:
            raise NameError('Please fit model first')
        class_names = np.unique(self.y)
        class_probabilities = {}
        for i in class_names:
            class_indices = np.where(self.y == i)[0]
            print (i, class_indices)
            try:
                class_samples = self.X[class_indices, :]
            except:
                class_samples = self.X[class_indices]
            class_prior = float(len(class_indices)) / len(self.y)
            kde = self.kerneldensityestimate(class_samples, x, gamma)
            class_probabilities[i] = kde * class_prior
        return max(class_probabilities, key=class_probabilities.get)









