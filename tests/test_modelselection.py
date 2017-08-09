import ML.modelselection as modelselection
import ML.regression as regression
import data
import numpy as np


def test_best_subset():
    X, y = data.tall_matrix_data_2()
    error_measure = modelselection.Error.mse
    model = regression.LinearRegression
    subset = modelselection.best_subset(X, y, model, 2, error_measure)
    assert 0 in subset
    assert 2 not in subset


def test_best_subset_forward():
    X, y = data.tall_matrix_data_2()
    error_measure = modelselection.Error.mse
    model = regression.LinearRegression
    subset = modelselection.best_subset(X, y, model, 2, error_measure,
                                        direction='forward')
    assert 0 in subset
    assert 2 not in subset


def test_best_subset_backward():
    X, y = data.tall_matrix_data_2()
    error_measure = modelselection.Error.mse
    model = regression.LinearRegression
    subset = modelselection.best_subset(X, y, model, 2, error_measure,
                                        direction='backward')
    assert 0 in subset
    assert 2 not in subset


def test_best_subset_combinatorial():
    X, y = data.tall_matrix_data_2()
    error_measure = modelselection.Error.mse
    model = regression.LinearRegression
    subset = modelselection.best_subset(X, y, model, 2, error_measure,
                                        direction='combinatorial')
    assert 0 in subset
    assert 2 not in subset


def test_error_mse():
    y = np.array([1, 2, 3, 4, 5])
    predictions = np.array([1.1, 2, 3.2, 4, 5])
    error = modelselection.Error.mse(y, predictions)
    assert np.isclose(error, 0.01)


def test_error_mean_classification_error():
    y = np.array([0, 0, 0, 1, 1, 1])
    predictions = np.array([0, 0, 1, 1, 1, 1])
    error = modelselection.Error.mean_classification_error(y, predictions)
    assert np.isclose(error, 0.166666)


def test_error_cross_entropy_error():
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    predictions = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    error = modelselection.Error.cross_entropy_error(y, predictions)
    assert np.isclose(error, 6.912757780650054)
