# Note: The perceptron requires linearly seperable data.
import numpy as np

class Perceptron:
    def __init__(self, max_iter = 100):
        self.max_iter = max_iter
        self.weights = np.NaN
        self.learned = False
        
    def predict(self, X):
        X = np.asarray(X)
        X = np.insert(X, 0, 1, axis = 1)
        if self.learned:
            prediction = np.sign(np.dot(X, np.transpose(self.weights)))
            return prediction
        else:
            raise NameError('Model has not been fit')
        
    def fit(self, X, y):
        y = np.asarray(y)
        X = np.insert(X, 0, 1, axis = 1)
        print (X)
        #Check if y contains only 1 or -1 values
        if False in np.in1d(y, [-1, 1]):
            raise NameError('y required to contain only 1 and -1')
        self.learned = True
        self.weights = np.zeros(np.shape(X)[1])
        # Update weights until they linearly separate inputs
        iteration = 0
        while not np.array_equal(np.sign(np.dot(X, np.transpose(self.weights))), y):
            if iteration > self.max_iter:
                print ('u done')
                return self
            classification = np.equal(np.sign(np.dot(X, np.transpose(self.weights))), y)
            misclassified_point = np.where(classification==False)[0][0]
            self.weights = self.weights + 0.001 * y[misclassified_point] * X[misclassified_point]
            iteration += 1
        return self
        
# h(x) = sign(transpose(w)*x)
# input weights and vector

### Algorithm
# Randomly assign w
# pick misclassified point (how to efficiently find?)
# update weight vecotr w = w + yn*Xn
# repeat until all are classified

from sklearn.datasets.samples_generator import make_blobs
blobs = make_blobs(n_samples = 10, n_features = 2, centers = 2, cluster_std = 0.0001)
X = blobs[0]
y = blobs[1]
y[y==0]=-1

a = Perceptron()
a.fit(X, y)
