"""
Tree based methods of learning (classification and regression)
"""
import abc
import numpy as np
import networkx as nx
from scipy.stats import mode

class BaseTree(object):
    """
    Base Tree for classification/regression.  Written for single variable/value
    binary split critereon.  Many methods needs to be rewritten if a more complex
    split critereon is desired.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """
        Attributes:
            graph (nx.DiGraph): Directed graph which stores tree
            nodes (int): Current number of nodes in tree
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]
            learned (bool): Keeps track of if model has been fit
        """
        self.graph = nx.DiGraph()
        self.graph.add_node(1)
        self.nodes = 1
        self.X = None
        self.y = None
        self.learned = False

    def fit(self, X, y, height, weights=None):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]
            height (int): height of tree
            weights (np.array): array of sample weights
                if None, all samples are weighted evenly

        Returns: an instance of self
        """
        self.X = X
        self.y = y
        self.weights = weights
        for layer in range(height):
            self.add_layer()
        self.compute_class_averages()
        self.learned = True
        return self

    def predict(self, x):
        """
        Args:
            x (np.array): Training data of shape[n_features,]

        Returns:
            prediction (float): predicted value

        Raises:
            ValueError if model has not been fit

        Notes:
            Currently, only a single data instance can be predicted at a time.
        """
        if not self.learned:
            raise NameError('Fit model first')
        current_node = 1
        leaves = self.get_leaves()
        while current_node not in leaves:
            children = self.graph.successors(current_node)
            current_variable = self.graph.node[current_node]['variable']
            current_cutoff = self.graph.node[current_node]['cutoff']
            if current_variable is None:
                return self.graph.node[current_node]['classval']
            if x[current_variable] > current_cutoff:
                current_node = children[1]
            else:
                current_node = children[0]
        return self.graph.node[current_node]['classval']

    def add_layer(self):
        """
        Used by Fit() to add a single layer at the bottom of the tree
        """
        leaves = self.get_leaves()
        for leaf in leaves:
            data_indices = self.partition_data(leaf)
            leaf_X = self.X[data_indices, :]
            leaf_y = self.y[data_indices]
            self.add_split(leaf, leaf_X, leaf_y)

    def get_leaves(self):
        """
        Used by add_layer() to get the leaves of the tree.
        """
        leaves = []
        for node in self.graph.nodes():
            if len(self.graph.successors(node)) == 0:
                leaves.append(node)
        return leaves

    def add_split(self, node_number, data, values):
        """
        Used by add_layer() to add two children at a leaf in the tree

        Args:
            node_number (int): Node in tree which a new split is added to
            data (np.ndarray): data of shape[n_samples, n_features]
                Data which node split will be based off of
            values (np.array): values of shape[n_samples,]
                Target values which node split will be based off of
        """
        min_feature, min_split = self.learn_split(data, values)
        self.graph.node[node_number]['variable'] = min_feature
        self.graph.node[node_number]['cutoff'] = min_split
        for i in range(2):
            self.nodes += 1
            self.graph.add_edge(node_number, self.nodes)

    def partition_data(self, node_number):
        """
        Partitions the training data at a given node.  Traverses the
        entire down to the indicated node.

        Args:
            node_number (int): Node in tree to partition data down to

        Returns:
            data_indices (np.array): Array of indices from training data which
                partition to node
        """
        predecessors = self.get_predecessors(node_number)
        predecessors.reverse()
        predecessors.append(node_number)
        data_indices = np.array(range(len(self.y)))
        node_count = 0
        while node_count < len(predecessors) - 1:
            current_node = predecessors[node_count]
            next_node = predecessors[node_count + 1]
            current_variable = self.graph.node[current_node]['variable']
            current_cutoff = self.graph.node[current_node]['cutoff']
            if current_cutoff is None:
                return []
            if next_node == min(self.graph.successors(current_node)):
                data_indices = data_indices[self.X[data_indices, current_variable] < current_cutoff]
            else:
                data_indices = data_indices[self.X[data_indices, current_variable] > current_cutoff]
            node_count += 1
        return data_indices

    def get_predecessors(self, node_number):
        """
        Used by parition_data() to get predecessors of a given node (to walk down the tree)
        """
        predecessors = []
        current_node = node_number
        while len(self.graph.predecessors(current_node)) > 0:
            current_node = self.graph.predecessors(current_node)[0]
            predecessors.append(current_node)
        return predecessors

    @abc.abstractmethod
    def compute_class_averages(self):
        """
        Method to compute average value for all nodes in the tree
        """
        return

    @abc.abstractmethod
    def learn_split(self, inputs, values):
        """
        Method to learn split given a data set (inputs) with target values (values)
        """
        return

class RegressionTree(BaseTree):
    """
    Regression Tree implimenting CART algorithm
    """

    def __init__(self):
        BaseTree.__init__(self)

    def learn_split(self, inputs, values):
        """
        CART algorithm to learn split at node in tree.
        Minimizes mean squared error of the two classes generated.

        Args:
            data (np.ndarray): data of shape[n_samples, n_features]
                Data which node split will be based off of
            values (np.array): values of shape[n_samples,]
                Target values which node split will be based off of

        Returns:
            min_split (float): feature value at which to split
            min_feature (int): feature number to split data by
                Essentially, the column number from the data which split is performed on
        """
        if self.weights == None:
            weights = np.ones(len(values))
        else:
            weights = np.array(self.weights)
        min_error = np.inf
        min_feature = None
        min_split = None
        for feature in range(np.shape(inputs)[1]):
            feature_vector = inputs[:, feature]
            sorted_vector = np.unique(np.sort(feature_vector))
            feature_splits = (sorted_vector[1:] + sorted_vector[:-1]) / 2
            for split in feature_splits:
                lower_class_average = np.mean(values[feature_vector < split])
                upper_class_average = np.mean(values[feature_vector > split])
                lower_class_errors = (values[feature_vector < split] - lower_class_average) * weights[feature_vector < split]
                upper_class_errors = (values[feature_vector > split] - upper_class_average) * weights[feature_vector > split]
                total_error = np.inner(lower_class_errors, lower_class_errors) + np.inner(upper_class_errors, upper_class_errors)
                if total_error < min_error:
                    min_error = total_error
                    min_feature = feature
                    min_split = split
        return min_feature, min_split

    def compute_class_averages(self):
        """
        Computes the class average of each node in the tree.
        Class average is mean of training data that partitions to the node.
        """
        for i in range(2, self.nodes + 1):
            parent = self.graph.predecessors(i)[0]
            if self.graph.node[parent]['cutoff'] is None:
                self.graph.node[i]['classval'] = self.graph.node[parent]['classval']
            else:
                node_indices = self.partition_data(i)
                classval = self.y[node_indices].mean()
                self.graph.node[i]['classval'] = classval

class ClassificationTree(BaseTree):
    """
    Classification Tree implimenting CART algorithm
    """

    def __init__(self):
        BaseTree.__init__(self)

    def learn_split(self, inputs, values):
        """
        CART algorithm to learn split at node in tree.
        Minimizes total misclassification error.

        Args:
            data (np.ndarray): data of shape[n_samples, n_features]
                Data which node split will be based off of
            values (np.array): values of shape[n_samples,]
                Target values which node split will be based off of

        Returns:
            min_split (float): feature value at which to split
            min_feature (int): feature number to split data by
                Essentially, the column number from the data which split is performed on
        """
        if self.weights == None:
            weights = np.ones(len(values))
        else:
            weights = np.array(self.weights)
        min_error = np.inf
        min_feature = None
        min_split = None
        for feature in range(np.shape(inputs)[1]):
            feature_vector = inputs[:, feature]
            sorted_vector = np.unique(np.sort(feature_vector))
            feature_splits = (sorted_vector[1:] + sorted_vector[:-1]) / 2
            for split in feature_splits:
                lower_class_mode = mode(values[feature_vector < split]).mode[0]
                upper_class_mode = mode(values[feature_vector > split]).mode[0]
                lower_class_errors = np.sum((values[feature_vector < split] != lower_class_mode).astype(int) * weights[feature_vector < split])
                upper_class_errors = np.sum((values[feature_vector > split] != upper_class_mode).astype(int) * weights[feature_vector > split])
                total_error = upper_class_errors + lower_class_errors
                if total_error < min_error:
                    min_error = total_error
                    min_feature = feature
                    min_split = split
        return min_feature, min_split

    def compute_class_averages(self):
        """
        Computes the class average of each node in the tree.
        Class average is the mode of training data that partitions to the node.
        """
        for i in range(2, self.nodes + 1):
            parent = self.graph.predecessors(i)[0]
            if self.graph.node[parent]['cutoff'] is None:
                self.graph.node[i]['classval'] = self.graph.node[parent]['classval']
            else:
                node_indices = self.partition_data(i)
                classval = mode(self.y[node_indices]).mode[0]
                self.graph.node[i]['classval'] = classval

class PrimRegression(BaseTree):
    """
    PRIM: Patient Rule Induction Method
    Decision at node peels of 10% of data which  maximizes response mean
    More "patient" than CART algorithm.

    NOTE:
        Since decision is a "box", many methods in BaseTree class are overwritten
        In the futute, BaseTree can be reworked to accomodate more flexible decisions
    """
    def __init__(self):
        BaseTree.__init__(self)

    def add_split(self, node_number, data, values):
        """
        Used by add_layer() to add two children at a leaf in the tree

        Args:
            node_number (int): Node in tree which a new split is added to
            data (np.ndarray): data of shape[n_samples, n_features]
                Data which node split will be based off of
            values (np.array): values of shape[n_samples,]
                Target values which node split will be based off of
        """
        cutoffs = self.learn_split(data, values)
        self.graph.node[node_number]['cutoffs'] = cutoffs
        for i in range(2):
            self.nodes += 1
            self.graph.add_edge(node_number, self.nodes)

    def learn_split(self, inputs, values):
        """
        PRIM algorithm to learn split at node in tree.
        Maximizes response mean after "boxing off" 90% of data.

        Args:
            data (np.ndarray): data of shape[n_samples, n_features]
                Data which node split will be based off of
            values (np.array): values of shape[n_samples,]
                Target values which node split will be based off of

        Returns:
            cutoffs (dict): Dictionary of cutoffs to use
            {variable: [min_cutoff, max_cutoff]}
            Example: {3, [-12.5, 10]} means samples boxed between 12.5 and 10
                on variable 3 are in the box.
            Note: in an early implimentation, this dictiory could contain
                single values.  Currently, it only ever contains a single
                value.  This can be simplified in the future.
        """
        target_bin_size = len(self.y)/10
        cutoffs = {}
        if len(values) <= target_bin_size:
            return cutoffs
        best_variable = None
        best_cutoff = [-np.inf, np.inf]
        mean_response = np.mean(values)
        for feature in range(np.shape(inputs)[1]):
            feature_vector = inputs[:, feature]
            sorted_vector = np.unique(np.sort(feature_vector))
            feature_splits = (sorted_vector[1:] + sorted_vector[:-1]) / 2
            lower_split, upper_split = [int(len(feature_splits) * 0.1), int(len(feature_splits) * 0.9)]
            boxed_data_upper = values[inputs[:, feature] > feature_splits[lower_split]]
            boxed_data_lower = values[inputs[:, feature] < feature_splits[upper_split]]
            max_split = max(np.mean(boxed_data_lower), np.mean(boxed_data_upper))
            if max_split > mean_response:
                mean_response = max_split
                if np.mean(boxed_data_upper) > np.mean(boxed_data_lower):
                    best_cutoff = [feature_splits[lower_split], np.inf]
                else:
                    best_cutoff = [-np.inf, feature_splits[upper_split]]
                best_variable = feature
        if best_variable is None:
            return cutoffs
        for i in range(np.shape(inputs)[1]):
            cutoffs[i] = [-np.inf, np.inf]
        cutoffs[best_variable] = best_cutoff
        return cutoffs

    def predict(self, x):
        """
        Args:
            x (np.array): Training data of shape[n_features,]

        Returns:
            prediction (float): predicted value

        Raises:
            ValueError if model has not been fit

        Notes:
            Currently, only a single data instance can be predicted at a time.
        """
        if not self.learned:
            raise NameError('Fit model first')
        current_node = 1
        leaves = self.get_leaves()
        while current_node not in leaves:
            children = self.graph.successors(current_node)
            if self.graph.node[current_node]['cutoffs'] is None:
                return self.graph.node[current_node]['classval']
            within_box = True
            for key in self.graph.node[current_node]['cutoffs']:
                current_variable = key
                current_cutoff = self.graph.node[current_node]['cutoffs'][key]
                if x[current_variable] < current_cutoff[0] or x[current_variable] > current_cutoff[1]:
                    within_box = False
            if within_box:
                current_node = children[0]
            else:
                current_node = children[1]
        return self.graph.node[current_node]['classval']

    def compute_class_averages(self):
        """
        Computes the class average of each node in the tree.
        Class average is the mean of training data that partitions to the node.
        """
        for i in range(2, self.nodes + 1):
            parent = self.graph.predecessors(i)[0]
            if self.graph.node[parent]['cutoffs'] == {}:
                self.graph.node[i]['classval'] = self.graph.node[parent]['classval']
            else:
                node_indices = self.partition_data(i)
                if len(node_indices) == 0:
                    self.graph.node[i]['classval'] = self.graph.node[parent]['classval']
                else:
                    classval = self.y[node_indices].mean()
                    self.graph.node[i]['classval'] = classval

    def partition_data(self, node_number):
        """
        Partitions the training data at a given node.  Traverses the
        entire down to the indicated node.

        Args:
            node_number (int): Node in tree to partition data down to

        Returns:
            data_indices (np.array): Array of indices from training data which
                partition to node
        """
        predecessors = self.get_predecessors(node_number)
        predecessors.reverse()
        predecessors.append(node_number)
        data = self.X
        data_indices = np.array(range(len(self.y)))
        node_count = 0
        while node_count < len(predecessors) - 1:
            temp_data = data[data_indices]
            current_node = predecessors[node_count]
            next_node = predecessors[node_count + 1]
            cutoff_dict = self.graph.node[current_node]['cutoffs']
            if cutoff_dict is None:
                return None
            in_box = self.partition_data_nodeless(temp_data, cutoff_dict)
            if in_box is None:
                return None
            if next_node == min(self.graph.successors(current_node)):
                data_indices = data_indices[in_box]
            else:
                data_indices = np.delete(data_indices, in_box)
            node_count += 1
            if len(data_indices) == 0:
                return []
        return data_indices

    @staticmethod
    def partition_data_nodeless(inputs, cutoff_dict):
        """
        Partitions inputs based off of a cutoff dictionary which can contain
        cutoffs for many varialbes (although this feature is currently unused)
        """
        data_indices = np.array(range(np.shape(inputs)[0]))
        if cutoff_dict is None:
            return None
        for key in cutoff_dict:
            current_variable = key
            current_cutoff_min = cutoff_dict[key][0]
            current_cutoff_max = cutoff_dict[key][1]
            boxed_data = data_indices[(inputs[data_indices, current_variable] < current_cutoff_max) & (inputs[data_indices, current_variable] > current_cutoff_min)]
            data_indices = boxed_data
        return data_indices

class DiscreteAdaBoost(object):
    """
    Ada Boost classifier.
    This implimentation produces a series of decisions stumps (decision trees with
    two terminal nodes).
    """
    def __init__(self):
        """
        Attributes:
            stump_count (int): current number of stumps generated for model
            stumps (list): list to hold stumps generated by moldel
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]
            weights (list): list of weights.  Each weight[i] is a list of weights
                containing sample weights of shape[n_samples, 1], corresponding to
                stump in stumps[i]
            alphas (list): List of alphas, which determine how much to weight each
                decision stump in final model.
            learned (bool): Keeps track of if model has been fit
        """
        self.stump_count = 0
        self.stumps = []
        self.X = None
        self.y = None
        self.weights = []
        self.alphas = []
        self.learned = False

    def fit(self, X, y, n_stumps=100):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.array): Target values of shape[n_samples]
            n_stumps (int): number of stumps in classifier

        Returns: an instance of self
        """
        self.X = X
        self.y = y
        n_samples = len(self.y)

        while self.stump_count < n_stumps:
            if len(self.weights) == 0:
                current_weights = np.ones(n_samples) / n_samples
            else:
                current_weights = self.weights[-1]
            self.add_stump(current_weights)
            self.stump_count += 1
        self.learned = True
        return self

    def add_stump(self, weights):
        """
        weights (np.array): array of weights of shape[n_samples,] to weight
            each sample in current stump

        Notes:
            Method adds a single decision stump to self.stumps and ALSO
            calculates weights of samples for the generated stump

        Returns: an instance of self.
            self.stumps and self.weights are appended with the newest stump/weights
        """
        stump = ClassificationTree()
        stump.fit(self.X, self.y, height=1, weights=weights)
        predictions = []
        for row in self.X:
            prediction = stump.predict(row)
            predictions.append(prediction)
        current_misclassifications = (predictions != self.y).astype(int)
        current_error = np.sum(current_misclassifications * weights) / np.sum(weights)
        current_alpha = np.log((1-current_error) / current_error)
        current_weights = weights * np.exp(current_alpha * current_misclassifications)
        self.weights.append(current_weights)
        self.alphas.append(current_alpha)
        self.stumps.append(stump)
        return self

    def predict(self, x):
        """
        Args:
            x (np.array): Training data of shape[n_features,]

        Returns:
            prediction (float): predicted value

        Raises:
            ValueError if model has not been fit

        Notes:
            Currently, only a single data instance can be predicted at a time.
        """
        if not self.learned:
            raise NameError('Fit model first')
        stump_predictions = []
        for i in range(self.stump_count):
            stump_prediction = self.stumps[i].predict(x)
            stump_predictions.append(stump_prediction)
        predictions = np.array(stump_predictions)
        prediction = np.sign(np.sum(predictions * self.alphas))
        return prediction

class GradientBoostingRegression(object):
    """
    Gradient boosting regression.
    A series of small trees (weak classifiers) combined to produce a
    hypothesis.  Each subsequent classifier fits a differential loss function
    of the previous model.  Currently, only MSE is supported, but the differential
    loss function can be a variety of functions.
    """
    def __init__(self):
        """
        Attributes:
            tree_count (int): current number of trees generated for model
            trees (list): list to hold tree generated by moldel
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.ndarray): Target values of shape[n_samples, 1]
            initial_hypothesis (float): Starting hypothesis for model.
                This is just mean of target values, once the model is fit.
            learning_rate (float): Contribution each subsequent tree makes to
                the final model.
            tree_depth (int): Depth of each tree generated by model.
            learned (bool): Keeps track of if model has been fit
        """
        self.tree_count = 0
        self.trees = []
        self.X = None
        self.y = None
        self.initial_hypothesis = None
        self.learning_rate = None
        self.tree_depth = None
        self.learned = False

    def fit(self, X, y, n_trees=20, tree_depth=3, learning_rate=0.1):
        """
        Args:
            X (np.ndarray): Training data of shape[n_samples, n_features]
            y (np.array): Target values of shape[n_samples]
            n_trees (int): number of trees in regressor
            tree_depth (int): height of each tree in regressor
            learning_rate (float): shrinkage parameter.  Determines how much
                of each new tree to contribute to final hypothesis.
        Returns: an instance of self
        """
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.tree_depth = tree_depth
        self.initial_hypothesis = np.mean(self.y)
        while self.tree_count < n_trees:
            current_predictions = []
            for row in self.X:
                current_predictions.append(self.predict(row))
            current_predictions = np.array(current_predictions)
            current_residuals = -(self.y - current_predictions)
            self.add_tree(current_residuals)
            self.tree_count += 1
        self.learned = True
        return self

    def add_tree(self, residuals):
        """
        residuals (np.array): array of residuals of shape[n_samples,] calculated
            from the current model

        Notes:
            Method adds a single decision tree to self.trees by fitting
            the residuals of the current model

        Returns: an instance of self.
        """
        tree = RegressionTree()
        tree.fit(self.X, residuals, self.tree_depth)
        self.trees.append(tree)

    def predict(self, x):
        """
        Args:
            x (np.array): Training data of shape[n_features,]

        Returns:
            prediction (float): predicted value

        Raises:
            ValueError if model has not been fit

        Notes:
            Currently, only a single data instance can be predicted at a time.
        """
        if not self.learned:
            raise NameError('Fit model first')
        prediction = self.initial_hypothesis
        for tree in self.trees:
            gradient = -tree.predict(x)
            prediction += self.learning_rate * gradient
        return prediction
