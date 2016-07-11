"""
Prototype methods are unstructured methods that represent that training
data with prototypes in the feature space.
"""
import random
import numpy as np
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

        Returns: an instance of self
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
            float: Returns predicted value

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

        Returns: an instance of self
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
                    current_distance = np.linalg.norm(self.cluster_centers[row, :] - self.samples[sample, :])
                    distances = np.append(distances, current_distance)
                nearest_center = distances.argsort()[0]
                self.sample_assignments[sample] = nearest_center
            #Update cluster centers to cluster mean
            new_centers = np.zeros((clusters, n_features))
            for cluster in range(clusters):
                samples_in_cluster = np.where(self.sample_assignments == cluster)[0]
                print(samples_in_cluster)
                new_centers[cluster, :] = self.samples[samples_in_cluster, :].mean(axis=0)
            if (self.cluster_centers == new_centers).all():
                break
            self.cluster_centers = new_centers
            iteration += 1
        self.learned = True
        return self

    def predict(self, x):
        """
        Note: currenly only works on single vector and not matrices

        Args:
            x (np.ndarray): Training data of shape[1, n_features]

        Returns:
            np.array: nearest cluster to x

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

class KMediods(object):
    """
    K mediods clustering.
    Similar to K means, however, datapoints (mediods/exemplars) are chosen
    as cluster centers.
    TODO: merge in with class Kmeans(), since a lot of repetative code.
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

    def fit(self, X, clusters=2, max_iter=20):
        """
        Randomly initializes clusers, iteratively update cluster center

        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            clusters (int): number of clusters to determine
            max_iter (int): maximum number of iterations through Lloyd's algorithm

        Returns: an instance of self
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
                distances = []
                for row in range(clusters):
                    difference = (self.cluster_centers[row, :] - self.samples[sample, :])
                    current_distance = np.linalg.norm(difference)
                    distances.append(current_distance)
                distances = np.array(distances)
                nearest_center = distances.argsort()[0]
                self.sample_assignments[sample] = nearest_center
            print(self.sample_assignments)
            # Update cluster centers to mediods that minimize cost
            # Cost is sum of distances to mediod (within group)
            new_centers = np.zeros((clusters, n_features))
            for cluster in range(clusters):
                cluster_cost = np.inf
                best_mediod = None
                for point in self.samples[self.sample_assignments == cluster, :]:
                    differences = self.samples[self.sample_assignments == cluster, :] - point
                    distances = np.dot(differences, differences.T)
                    total_distance = np.trace(distances)
                    if total_distance < cluster_cost:
                        cluster_cost = total_distance
                        best_mediod = point
                    new_centers[cluster, :] = best_mediod
            if (self.cluster_centers == new_centers).all():
                break
            self.cluster_centers = new_centers
            iteration += 1
        self.learned = True
        return self

    def predict(self, x):
        """
        Note: currenly only works on single vector and not matrices

        Args:
            x (np.ndarray): Training data of shape[1, n_features]

        Returns:
            np.array: nearest cluster to x

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

    def fit(self, X, y, n_prototypes=5, epsilon=0.01, max_iter=1000):
        """
        Randomly initializes clusers, uses LLyod's algorithm to find optimal clusters

        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.array): Target values of shape[n_samples]
            n_prototypes (int): number of prototypes per class
            epsilon (float): learning rate.  How much to move each prototype per iteration
            max_iter (int): maximum number of iterations through Lloyd's algorithm

        Returns: an instance of self
        """
        self.X = X
        self.y = y
        n_samples = len(self.y)
        class_vals = np.unique(self.y)
        class_indices = {}
        for val in class_vals:
            class_indices[val] = np.where(self.y == val)[0]
        # Initialize prototypes with random data points
        for val in class_vals:
            random_class_indices = np.random.choice(class_indices[val], n_prototypes, replace=False)
            self.prototypes[val] = self.X[random_class_indices, :]
        """
        LVQ Algorithm:
            1) Pick random data point (with replacement)
            2) Identify closest prototype (2 norm)
            3) If same class, move protytpe toward training point
            4) If different class, move prototype away from training point
            5) Repeat
        """
        iterations = 0
        while iterations < max_iter:
            current_index = np.random.choice(np.arange(n_samples))
            current_data = self.X[current_index, :]
            # Find closest prototype
            closest_prototype = [None, None, None]
            closest_distance = np.inf
            for key in self.prototypes:
                for index, row in enumerate(self.prototypes[key]):
                    distance = np.linalg.norm(row - current_data)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_prototype = [key, index, row]
            print(closest_prototype)
            if closest_prototype[0] == self.y[current_index]:
                self.prototypes[closest_prototype[0]][closest_prototype[1]] += epsilon * (current_data - closest_prototype[2])
            else:
                self.prototypes[closest_prototype[0]][closest_prototype[1]] -= epsilon * (current_data - closest_prototype[2])
            iterations += 1
        self.learned = True
        return self

    def predict(self, x):
        """
        Note: currenly only works on single vector (one data instance) and not matrices

        Args:
            x (np.array): sample data of shape[n_features]

        Returns:
            predicted class of sample

        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
        closest_prototype = None
        closest_distance = np.inf
        for key in self.prototypes:
            for row in self.prototypes[key]:
                distance = np.linalg.norm(row - x)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_prototype = key
        prediction = closest_prototype
        return prediction

class DANN(object):
    """
    Discriminant Adaptive Nearest Neighbors (DANN).
    DANN adaptively elongates neighborhoods along boundry regions.
    Useful for high dimensional data.
    Reference:
        Hastie, Trevor, and Robert Tibshirani.
        "Discriminant adaptive nearest neighbor classification."
        IEEE transactions on pattern analysis and machine intelligence
        18.6 (1996): 607-616.
    """
    def __init__(self):
        """
        Attributes:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.array): Target values of shape[n_samples]
            neighborhood_size (int): number of nearest neighbors to consider when predicting
            learned (bool): Keeps track of if model has been fit
        """
        self.X = None
        self.y = None
        self.neighborhood_size = None
        self.learned = False

    def fit(self, X, y, neighborhood_size=50, epsilon=1):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.array): Target values of shape[n_samples]
            neighborhood_size (int): number of nearest neighbors to consider when predicting
            epsilon (float): learning rate.  How much to move each prototype per iteration

        Returns: an instance of self
        """
        self.X = X
        self.y = y
        self.neighborhood_size = neighborhood_size
        self.epsilon = epsilon
        self.learned = True
        return self

    def predict(self, x, k=10):
        """
        Args:
            x1 (np.array): query point of shape[n_features]
            k (int): number of nearest neighbors to consider

        Returns: 
            Predicted class of sample

        Raises:
            ValueError if model has not been fit
        """
        if not self.learned:
            raise NameError('Fit model first')
        n_features = len(x)
        distances = []
        for row in self.X:
            distance = np.linalg.norm(row-x)
            distances.append(distance)
        distances = np.array(distances)
        nearest_neighbors = np.argsort(distances)[:self.neighborhood_size]
        neighborhood_X = self.X[nearest_neighbors, :]
        neighborhood_X_mean = neighborhood_X.mean(axis=0)
        neighborhood_y = self.y[nearest_neighbors]
        neighborhood_classes = np.unique(neighborhood_y)
        class_frequencies = {}
        within_class_cov = np.zeros((n_features, n_features))
        between_class_cov = np.zeros((n_features, n_features))
        for target_class in neighborhood_classes:
            class_indices = np.where(neighborhood_y == target_class)[0]
            class_frequencies[target_class] = np.sum(neighborhood_y == target_class) / self.neighborhood_size
            class_covariance = np.cov(neighborhood_X[class_indices, :], rowvar=False)
            within_class_cov += class_covariance * class_frequencies[target_class]
            class_mean = neighborhood_X[class_indices, :].mean(axis=0)
            between_class_cov += np.outer(class_mean - neighborhood_X_mean, class_mean - neighborhood_X_mean) * class_frequencies[target_class]
        # W* = W^-.5
        # B* = W*BW*
        W_star = np.linalg.pinv(np.nan_to_num(np.power(within_class_cov, 0.5)))
        B_star = np.dot(W_star, between_class_cov).dot(W_star)
        I = np.identity(n_features)
        sigma = W_star.dot(B_star + self.epsilon * I).dot(W_star)
        distances = []
        for row in self.X:
            distances.append(self.DANN_distance(x, row, sigma))
        distances = np.array(distances)
        nearest = distances.argsort()[:k]
        prediction = stats.mode(self.y[nearest]).mode[0]
        return prediction

    def DANN_distance(self, x0, x1, sigma):
        """
        Computes the distance between x0 and x1 using the DANN metric
        which is adaptively defined at query locus
        Args:
            x1 (np.array): query point of shape[n_features]
            x2 (np.array): reference point of shape[n_features]
            sigma (np.ndarray): array of shape[n_features, n_features]
        """
        difference = x0 - x1
        distance = difference.T.dot(sigma).dot(difference)
        return distance
