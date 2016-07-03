"""Module containing fucntions for validating models and model selection"""

import numpy as np

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
