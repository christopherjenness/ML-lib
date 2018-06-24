"""
Kernel methods estimates the target function by fitting seperate
functions at each point using local smoothing of training data.
"""

import numpy as np


class KernelMethods(object):
    """
    Kernel methods for classification and estimation
    """
    def __init__(self):
        """
        Attributes:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            X_intercept (np.ndarray): Training data of shape
                [n_samples, n_features + 1]
                Adds a column of ones to training data
            y (np.ndarray): Target values of shape[n_samples, 1]
            learned (bool): Keeps track of if model has been fit
        """
        self.X = np.NaN
        self.X_intercept = np.NaN
        self.y = np.NaN
        self.learned = False

    def fit(self, X, y):
        """
        For Kernel Methods, data is stored in memory and fitting is done at
        prediction time.

        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]
        """
        self.X = np.asarray(X)
        self.X_intercept = np.column_stack((np.ones(np.shape(X)[0]), X))
        self.y = y
        self.learned = True

    @staticmethod
    def epanechnikovkernel(x0, x, gamma):
        """Epanechnikov Kernel"""
        t = np.linalg.norm(x - x0) / gamma
        if t > 1:
            return 0
        return 0.75 * (1 - t**2)

    @staticmethod
    def tricubekernel(x0, x, gamma):
        """Tricube Kernel"""
        t = np.linalg.norm(x - x0) / gamma
        if t > 1:
            return 0
        return (1 - t**3)**3

    @staticmethod
    def gaussiankernel(x0, x, gamma):
        """Gaussian Kernel"""
        t = -np.linalg.norm(x - x0)**2 / 2 * (gamma**2)
        return np.exp(t)

    def nadarayaaverage(self, x, kernel, gamma):
        """
        Regression estimate of the target value by weighted averaging
        of nearby training examples

        Args:
            x (np.array): Training data of shape[n_features,]
            kernel (function): {epanechinokovkernel, tricubekernel}
                kernel used to weight training examples
            gamma (float): parameter used for kernel
        Notes:
            Currently can only predict a single data instance.
        """
        x = np.array(x)
        if not self.learned:
            raise NameError('Please fit model first')
        numerator = 0
        denominator = 0
        for row in range(np.shape(self.X)[0]):
            numerator += kernel(self.X[row], x, gamma) * self.y[row]
            denominator += kernel(self.X[row], x, gamma)
        return numerator / denominator

    def locallinearregression(self, x, kernel, gamma):
        """
        Local linear regression eliminates bias at boundries
        of domain.  It uses weighted least squares, determining
        weights from the kernel.

        Args:
            x (np.array): Training data of shape[1, n_features]
                note: currently only single samples can be predicted
                at a time.
            kernel (function): {epanechinokovkernel, tricubekernel}
                kernel used to weight training examples
            gamma (float): parameter used for kernel
        """
        W = []
        for row in range(np.shape(self.X)[0]):
            W.append(kernel(self.X[row], x, gamma))
        W = np.diag(W)
        # Calculate solution using closed form solution
        solution_a = np.linalg.pinv(self.X_intercept.T.dot(W)
                                    .dot(self.X_intercept))
        solution_b = self.X_intercept.T.dot(W).dot(self.y)
        solution = np.dot(solution_a, solution_b)
        x = np.asarray(x)
        x = np.column_stack((1, x))
        prediction = np.inner(x, solution)
        return float(prediction)

    @staticmethod
    def logistic_function(logistic_input):
        """
        Args:
            logistic_input (np.ndarray): array of shape[n_samples, 1]

        Returns:
            np.ndarray: shape[n_samples, 1], logistic transformation of data
        """
        return 1 / (1 + np.exp(-logistic_input))

    def locallogisticHessian(self, theta, weights, reg_param):
        """
        Hessian for regulatrized local logistic regression L2 loss

        Args:
            theta (np.array): Current lwlr parameters of shape
                [1, n_features]
            weights (np.array): training set weights of shape
                [n_samples, 1]
            reg_param (float): L2 regularization weight. If 0, no
                no regulatrization is used.

        Returns:
            Hessian (np.ndarray): Hessian of shape [n_features, n_features]
        """
        # Add bias to X
        X = np.insert(self.X, 0, 1, axis=1)
        
        D = []
        for row in range(np.shape(X)[0]):
            D.append(weights[row] *
                     self.logistic_function(np.dot(X[row, :],
                                                   np.transpose(theta))) *
                     (1 -
                      self.logistic_function(np.dot(X[row, :],
                                                    np.transpose(theta)))))
        D = np.diag(D)
        hessian = (np.matmul(np.matmul(X.T, D),
                             X) -
                   np.identity(np.shape(X)[1]) * reg_param)
        return hessian

    def locallogisticregression(self, x, kernel, gamma, reg_param=0,
                                iterations=10, alpha=0.001):
        """
        Local linear logistic eliminates bias at boundries
        of domain.  It uses weighted least squares, determining
        weights from the kernel.

        Args:
            x (np.array): Training data of shape[1, n_features]
                note: currently only single samples can be predicted
                at a time.
            kernel (function): {epanechinokovkernel, tricubekernel}
                kernel used to weight training examples
            gamma (float): parameter used for kernel
            reg_param (float): L2 regularization weight
                if 0, no regulatization is prefermed
            iterations (int): number of gradient descent steps to take
            alpha (float): depth of each gradient descent step to take
        """
        # Add bias to X
        X = np.insert(self.X, 0, 1, axis=1)
        x = np.insert(x, 0, 1)

        # Set training set weights for query points
        W = [kernel(X[row], x, gamma)
             for row in range(np.shape(X)[0])]

        # Initialize theta
        theta = np.zeros(np.shape(X)[1]) + 0.0001

        # Newtons Method
        iteration = 0
        while iteration < iterations:
            hessian = self.locallogisticHessian(theta, W, reg_param)
            z = [W[row] * (self.y[row] -
                 self.logistic_function(np.dot(X[row, :],
                                               theta)))
                 for row in range(np.shape(X)[0])]
            gradient = np.matmul(X.T, z) - (reg_param * theta)
            step_direction = -np.dot(np.linalg.pinv(hessian), gradient)
            theta = theta + alpha * step_direction
            iteration += 1
        prediction = self.logistic_function(np.dot(x, theta))
        return prediction

    def kerneldensityestimate(self, x, gamma):
        """
        Provides a gaussian kernel density at point given a sample.
        If the KDE of the entire sample space is required, this method can
        easily be augmented.

        Args:
            x (np.array): Test data of shape [n_features]
            gamma: gaussian width from which to sample

        Returns:
            float: KDE estimate at point x, given samples
        """
        samples = np.matrix(self.X)
        N = len(x)
        estimate = 0
        for row in range(np.shape(samples)[0]):
            x_i = np.array(x) - samples[row, :]
            gaussiankernel = ((1/(2*np.pi)**0.5 * gamma**2)**N *
                              np.exp(-np.linalg.norm(x_i)**2 / (2 * gamma**2)))
            estimate += gaussiankernel
        estimate /= N
        return estimate

    def kerneldensitypredict(self, x, gamma):
        """
        Kernel density calssification prediction based on KDE of each class

        x (np.array): Test data of shape [n_features]
        gamma: gaussian width from which to sample in KDE

        Returns: float: Returns predicted class of test data x

        Raises:
            ValueError if model has not been fit

        Note:
            Can only predict single data point, currently
        """
        if not self.learned:
            raise NameError('Please fit model first')
        class_names = np.unique(self.y)
        class_probabilities = {}
        for i in class_names:
            class_indices = np.where(self.y == i)[0]
            class_prior = float(len(class_indices)) / len(self.y)
            kde = self.kerneldensityestimate(x, gamma)
            class_probabilities[i] = kde * class_prior
        prediction = max(class_probabilities, key=class_probabilities.get)
        return prediction
