"""Useful algorithms for selecting models"""

import numpy as np
import itertools

def best_subset(X, y, model, parameters, error_measure, direction='forward'):
    """
    Function for selecting a subset of parametrs from X, which minimize the in sample
    error measure.  Algorithm acts in a greedy manner, either adding one parameter at
    a time (forward), or removing one at a time (backward).

    Args:
        X (np.ndarray): Training data of shape[n_samples, n_features]
        y (np.ndarray): Target values of shape[n_samples, 1]
        model (class): Machine Learning model containing fit and predict methods
            For example LinearRegression from linearregression module
        error_measure (function): function(y, predictions) for measuring the model fit.
            For example, MSE error from error module
        parameters (int): number of parameters in subset to return
        direction: {'forward', 'backward', 'combinatorial'}
            forward adds one parameter at a time in a greedy manner
            backwards removes one parameter at a time in a greedy manner
            combinatorial tries all combinations of parameters
                combinatorial is not reccomended for large number of parameters

    Returns:
        np.ndarray: shape[parameters, 1], best subset of parameters that minimize in sample error
    """

    n_features = np.shape(X)[1]
    n_samples = np.shape(X)[0]
    if direction not in ['forward', 'backward', 'combinatorial']:
        raise  NameError("direction must be 'forward', 'backward', or 'combinatorial'")
    if direction == 'forward':
        param_count = 0
        current_params = []
        # Add one parameter at a time
        # Save parameter than minimizes error_measure
        while param_count < parameters:
            best_param = np.inf
            best_error = np.inf
            # Loop through remaining feature to determine best
            for feature in range(n_features):
                # only look at features not already selected
                if feature not in current_params:
                    test_column = X[:, feature]
                    if len(current_params) > 0:
                        test_array = np.column_stack((X[:, current_params], test_column))
                    else:
                        test_array = test_column
                    classifier = model()
                    classifier.fit(test_array, y)
                    test_error = error_measure(y, classifier.predict(test_array))
                    if test_error < best_error:
                        best_param, best_error = feature, test_error
            current_params.append(best_param)
            param_count += 1
    if direction == 'backward':
        param_count = n_features
        current_params = list(range(n_features))
        # Remove one parameter at a time
        # Save remaining parameters than minimizes error_measure
        while param_count > parameters:
            worst_param = np.inf
            worst_error = np.inf
            # Loop through remaining feature to determine least predictive
            for feature in range(n_features):
                if feature in current_params:
                    test_array = np.delete(X, feature, 1)
                    classifier = model()
                    classifier.fit(test_array, y)
                    test_error = error_measure(y, classifier.predict(test_array))
                    if test_error < worst_error:
                        worst_param, worst_error = feature, test_error
            current_params.remove(worst_param)
            param_count -= 1
    if direction == 'combinatorial':
        all_params = list(range(n_features))
        best_params = np.inf
        best_error = np.inf
        for param_subset in itertools.combinations(all_params, parameters):
                test_array = X[:, param_subset]
                classifier = model()
                classifier.fit(test_array, y)
                test_error = error_measure(y, classifier.predict(test_array))
                if test_error < best_error:
                    best_params, best_error = param_subset, test_error
        current_params = list(best_params)
    return sorted(current_params)

def test_train_splitter(X, y, test_fraction=0.2, randomize=True):
    """
    Splits X and y arrays into test and train sets

    Args:
        X (np.ndarray): Training data of shape[n_samples, n_features]
        y (np.ndarray): Target values of shape[n_samples, 1]
        test_fraction (float): fraction of data to partition into test set
            Must be in range [0, 1]
        randomize (bool): optional shuffling of data prior to segregation
            highly recommended for most validation procedures

    Returns:
        Returns 4 arrays: X_train, X_test, y_train, y_test
        X_train (np.ndarray):
            Training data of shape[n_samples * test_fraction, n_features]
        X_test (np.ndarray):
            Testing data of shape[n_samples * (1 - test_fraction), n_features]
        y_train (np.ndarray):
            Target Values of shape[n_samples * test_fraction, 1]
        y_test (np.ndarray):
            Target Values of shape[n_samples * (1 - test_fraction), 1]
    """
    assert 0 <= test_fraction <= 1
    aggregate = np.column_stack((X, y))
    if randomize:
        np.random.shuffle(aggregate)
    split_index = int(len(y) * test_fraction)
    X_train = aggregate[split_index:, :2]
    X_test = aggregate[0:split_index, :2]
    y_train = aggregate[split_index:, 2]
    y_test = aggregate[0:split_index, 2]
    return X_train, X_test, y_train, y_test

def k_fold_generator(data_length, folds=10, randomize=True):
    """
    generator of indices to split data of given length into test and train sets.
    Useful for K-fold cross validation.

    Args:
        data_length (int): Length of data to be split into test and training sets
        folds (int): number of splits to make in the data.
        randomize (bool): Option to shuffle indices prior to splitting

    Yields:
        train_indices, test_indices
            train_indices (np.array): array containing indices for train set
            test_indices (np.array): array containing indices for test set
    """
    indices = np.arange(data_length)
    if randomize:
        np.random.shuffle(indices)
    step_size = int(data_length / folds)
    current_position = 0
    while True:
        next_position = current_position + step_size
        train_indices = np.append(indices[:current_position], indices[next_position:])
        test_indices = indices[current_position: next_position]
        yield train_indices, test_indices
        current_position += step_size
        if current_position >= data_length:
            return

class Error(object):
    """
    Common error measures for model predictions when true answer is known
    """
    @staticmethod
    def mse(y, predictions):
        """Mean Square Error"""
        n_samples = len(y)
        differences = y - predictions
        return np.dot(differences, differences.T) / n_samples

    @staticmethod
    def mean_classification_error(y, predictions):
        """Average classiication error"""
        n_samples = len(y)
        correct_classifications = np.equal(y, predictions)
        return (n_samples - correct_classifications.sum()) / n_samples

    @staticmethod
    def cross_entropy_error(y, predictions):
        """Cross Entropy Error"""
        n_samples = len(y)
        predictions[predictions==0] = 0.00000001
        predictions[predictions==1] = 0.99999999
        cross_entropy_errors = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
        return np.sum(cross_entropy_errors)
