import numpy as np
import error
import linearregression

def best_subset(X, y, model, parameters, error_measure, direction='forward'):
    n_features = np.shape(X)[1]
    n_samples = np.shape(X)[0]
    if direction not in ['forward', 'backward']:
        raise  NameError("direction must be 'forward' or 'backward'")
    if direction=='forward':
        param_count = 0
        current_params = []
        # Add one parameter at a time
        # Save parameter than minimizes error_measure
        while param_count < parameters:
            best_param = np.inf
            best_error = np.inf
            # Loop through remaining feature to determine best
            for feature in range(n_features):
                if feature not in current_params:
                    test_column = X[:,feature]
                    if len(current_params) > 0:
                        test_array = np.column_stack((X[:,current_params], test_column))
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
    if direction =='backward':
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