"""Useful algorithms for selecting models"""

import numpy as np

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
        direction: {'forward', 'backward'}
            forward adds one parameter at a time in a greedy manner
            backwards removes one parameter at a time in a greedy manner

    Returns:
        current_params (np.ndarray): shape[parameters, 1]
        Returns best subset of parameters that minimize in sample error
    """

    n_features = np.shape(X)[1]
    n_samples = np.shape(X)[0]
    if direction not in ['forward', 'backward']:
        raise  NameError("direction must be 'forward' or 'backward'")
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
        return current_params
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
        return current_params
