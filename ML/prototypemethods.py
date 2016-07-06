"""
Prototype methods are unstructured methods that represent that training
data with prototypes in the feature space.
"""
import numpy as np
import random
from scipy import stats

class KNearestNeighbor(object):
    """
    K nearest neighbors classification and regression
    """
    def __init__(self):
        """
        Attributes:
            samples (np.ndarray): Data of known target values
            values (np.ndarray): Known target values for data
            learned (bool): Keeps track of if model has been fit
        """
        self.samples = np.nan
        self.values = np.nan
        self.learned = False

    def fit(self, X, y):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]

        Returns:
            self: Returns an instance of self
        """
        self.samples = X
        self.values = y
        self.learned = True
        return self

    def predict(self, x, k=1, model='regression'):
        """
        Note: currenly only works on single vector and not matrices

        Args:
            x (np.ndarray): Training data of shape[1, n_features]
            k (int): number of nearest neighbor to consider
            model: {'regression', 'classification'}
                K nearest neighbor classification or regression.
                Choice most likely depends on the type of data the model was fit with.

        Returns:
            prediction (float): Returns predicted value

        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
        distances = np.array([])
        for row in range(np.shape(self.samples)[0]):
            #add distance from x to sample row to distances vector
            distances = np.append(distances, np.linalg.norm(x - self.samples[row, :]))
            nearestneighbors = distances.argsort()[:k]
        if model == 'regression':
            prediction = self.values[nearestneighbors].mean()
        if model == 'classification':
            prediction = stats.mode(self.values[nearestneighbors]).mode
        return prediction
        
class KMeans(object):
    """
    K means clustering
    """
    def __init__(self):
        """
        Attributes:
            samples (np.ndarray): Data to be clusters
            sample_assignments (np.array): cluster assignments for each sample
            cluster_centers (np.ndarray): coodinantes of cluster centers
            learned (bool): Keeps track of if model has been fit
        """
        self.samples = np.nan
        self.sample_assignments = np.nan
        self.cluster_centers = np.nan
        self.learned = False

    def fit(self, X, clusters=2, max_iter=1000):
        """
        Randomly initializes clusers, uses LLyod's algorithm to find optimal clusters
        
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            clusters (int): number of clusters to determine
            max_iter (int): maximum number of iterations through Lloyd's algorithm

        Returns:
            self: Returns an instance of self
        """
        self.samples = X
        n_samples, n_features = np.shape(X)
        #Initialize centers to random data points
        self.cluster_centers = X[random.sample(range(n_samples), clusters), :]
        #Initialize sample assignmnets to zero
        self.sample_assignments = np.zeros(n_samples)
        iteration = 1
        while iteration < max_iter:
            print(iteration)
            #Assign each point to nearest center
            for sample in range(n_samples):
                distances = np.array([])
                for row in range(clusters):
                    current_distance =  np.linalg.norm(self.cluster_centers[row, :] - self.samples[sample, :])
                    distances = np.append(distances, current_distance)
                nearest_center = distances.argsort()[0]
                self.sample_assignments[sample] = nearest_center
            #Update cluster centers to cluster mean
            new_centers = np.zeros((clusters, n_features))
            for cluster in range(clusters):
                samples_in_cluster = np.where(self.sample_assignments == cluster)[0]
                print(samples_in_cluster)
                new_centers[cluster, :] = self.samples[samples_in_cluster, :].mean(axis = 0)
            print(self.cluster_centers)
            print(new_centers)
            if (self.cluster_centers == new_centers).all():
                break
            self.cluster_centers = new_centers
            iteration += 1
            print(iteration)
        self.learned = True  
        return self

    def predict(self, x):
        """
        Note: currenly only works on single vector and not matrices

        Args:
            x (np.ndarray): Training data of shape[1, n_features]

        Returns:
            prediction (np.array): Returns nearest cluster to x

        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
        distances = np.array([])
        for row in range(np.shape(self.cluster_centers)[0]):
            #append distance from x to cluster_centers row into distances vector
            distances = np.append(distances, np.linalg.norm(x - self.cluster_centers[row, :]))
        nearestneighbor = distances.argsort()[0]
        prediction = self.cluster_centers[nearestneighbor, :]
        return prediction
        
class LearningVectorQuantization(object):
    """
    Learning Vector Quantization: Prototypes are attracted to training 
    points of correct class, and repeled from training points in incorrect
    classes.
    """
    def __init__(self):
        """
        Attributes:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.array): Target values of shape[n_samples]
            prototypes (dict): location of learned protytpes for each class
            learned (bool): Keeps track of if model has been fit
        """
        self.X = None
        self.y = None
        self.prototypes = {}
        self.learned = False

    def fit(self, X, y, prototypes=5, max_iter=1000):
        """
        Randomly initializes clusers, uses LLyod's algorithm to find optimal clusters
        
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.array): Target values of shape[n_samples]
            prototypes (int): number of prototypes per class
            max_iter (int): maximum number of iterations through Lloyd's algorithm

        Returns:
            self: Returns an instance of self
        """
        self.learned = True
        return self

    def predict(self, x):
        """
        Note: currenly only works on single vector (one data instance) and not matrices

        Args:
            x (np.array): sample data of shape[n_features]

        Returns:
            prediction: Returns predicted class of sample

        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
        return
            
            
            
            
            
            
