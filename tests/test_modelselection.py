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


def test_k_fold_generator():
    np.random.seed(10)
    splitter = modelselection.k_fold_generator(20)
    split = splitter.next()
    assert len(split[0]) == 18
    assert len(split[1]) == 2


def test_k_fold_generator_odd():
    np.random.seed(10)
    splitter = modelselection.k_fold_generator(21)
    split = splitter.next()
    assert len(split[0]) == 19
    assert len(split[1]) == 2


def test_test_train_splitter():
    X, y = data.categorical_2Dmatrix_data_big()
    X_train, X_test, y_train, y_test = modelselection.test_train_splitter(X, y)
    assert X_train.shape == (9, 2)
    assert X_test.shape == (2, 2)
    assert len(y_train) == 9
    assert len(y_test) == 2
