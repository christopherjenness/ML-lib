#Note: The perceptron requires linearly seperable data.
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, max_iter = 100, learning_rate=1, pocket = False):
        """
        Args: 
            max_iter (int): Maximum number of iterations through PLA before stoping
            learning_rate (float): PLA step size
            pocket (bool): 
                
        Attributes::
            weights (np.ndarray): vector of weights for linear separation
            learned (bool): Keeps track of if perceptron has been fit
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.pocket = pocket
        
        self.weights = np.NaN
        self.learned = False
        
    def predict(self, X):
        """
        Args: 
            X (np.ndarray): Training data of shape[n_samples, n_features]
                
        Returns:
            self: Returns an instance of self
        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
        # Add column of 1s to X for perceptron threshold
        X = np.asarray(X)
        if X.ndim==1:
            X = np.insert(X, 0, 1, axis = 1)
        else:
            X = np.insert(X, 0, 1, axis = 1)
        prediction = np.sign(np.dot(X, np.transpose(self.weights)))
        return prediction
        
    def fit(self, X, y):
        """
        Args: 
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1] 
                (vals must be -1 or 1)
                
        Returns:
            prediction (np.ndarray): shape[n_samples 1)
                Returns predicted values
        Raises:
            ValueError if y contains values other than -1 or 1
        """
        y = np.asarray(y)
        X = np.insert(X, 0, 1, axis = 1)
        #Check if y contains only 1 or -1 values
        if False in np.in1d(y, [-1, 1]):
            raise NameError('y required to contain only 1 and -1')
        self.weights = np.zeros(np.shape(X)[1])
        # Update weights until they linearly separate inputs
        iteration = 0
        while not np.array_equal(np.sign(np.dot(X, np.transpose(self.weights))), y):
            if iteration > self.max_iter:
                return self
            classification = np.equal(np.sign(np.dot(X, np.transpose(self.weights))), y)
            misclassified_point = np.where(classification==False)[0][0]
            self.weights = self.weights + self.learning_rate * y[misclassified_point] * X[misclassified_point]
            iteration += 1
        self.learned = True
        return self
        
# h(x) = sign(transpose(w)*x)
# input weights and vector

### Algorithm
# Randomly assign w
# pick misclassified point (how to efficiently find?)
# update weight vecotr w = w + yn*Xn
# repeat until all are classified
###
from sklearn.datasets.samples_generator import make_blobs
blobs = make_blobs(n_samples = 10, n_features = 2, centers = 2, cluster_std = 0.0001)
X = blobs[0]
y = blobs[1]
y[y==0]=-1

a = Perceptron()
a.fit(X, y)

### 

a = np.random.uniform(-1, 1, 100)
b = np.random.uniform(-1, 1, 100)
X = np.column_stack((a,b))
r1 = np.random.uniform(-1, 1, 2)
r2 = np.random.uniform(-1, 1, 2)
decision_vector = [r2[0]-r1[0],r2[1] - r2[0]]   # Vector 1
point_vectors1, point_vectors2 = r2[0]-X[:,0], r2[1]-X[:,1]   # Vector 1
point_vectors = np.column_stack((point_vectors1,point_vectors2))
xp = decision_vector[1]*point_vectors[:,1] - decision_vector[1]*point_vectors[:,0]
y = np.sign(xp)


perc = Perceptron(learning_rate = 1)
#how_many.append(perc.fit(X, y))
perc.fit(X, y)

perc.predict(X) - y






