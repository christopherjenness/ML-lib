import ML.descentmethods as descentmethods
import ML.regression as regression
import data
import numpy as np
import pytest


@pytest.fixture
def gradient():
    def grad(X, y, weights):
        hypothesis = np.dot(X, weights) - y
        gradient = np.dot(np.transpose(X), hypothesis) / np.size(y)
        return gradient
    return grad


@pytest.fixture
def hessian():
    def hess(X, weights):
        hessian = np.matmul(X.T, X)
        return hessian
    return hess


def test_gradientdescent():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.gradientdescent(X, y, gradient())
    target = [0.47, 0.84]
    np.testing.assert_array_almost_equal(weights, target, 1)

def test_gradientdescent_alpha():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.gradientdescent(X, y, gradient(), alpha=0.001)
    target = [0.47, 0.84]
    np.testing.assert_array_almost_equal(weights, target, 1)


def test_gradientdescent_lowiterations():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.gradientdescent(X, y, gradient(), iterations=2)
    target = [0.1, 0.4]
    np.testing.assert_array_almost_equal(weights, target, 1)


def test_gradientdescent_initialweights():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.gradientdescent(X, y, gradient(),
                                             initial_weights=np.array([0.2, 0.2]))
    target = [0.47, 0.84]
    np.testing.assert_array_almost_equal(weights, target, 1)


def test_gradientdescent_stochastic():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.gradientdescent(X, y, gradient(),
                                             stochastic=True)
    target = [0.47, 0.84]
    np.testing.assert_array_almost_equal(weights, target, 1)


def test_gradientdescent_regparam():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.gradientdescent(X, y, gradient(),
                                             reg_param=0.01)
    target = [0.47, 0.84]
    np.testing.assert_array_almost_equal(weights, target, 1)


def test_steepestdescent():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.steepestdescent(X, y, gradient())
    target = [0.47, 0.84]
    np.testing.assert_array_almost_equal(weights, target, 1)


def test_steepestdescent_alpha():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.steepestdescent(X, y, gradient(),
                                             alpha=0.001)
    target = [0.47, 0.84]
    np.testing.assert_array_almost_equal(weights, target, 1)


def test_steepestdescent_lowiterations():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.steepestdescent(X, y, gradient(),
                                             iterations=2)
    target = [0, 0.4]
    np.testing.assert_array_almost_equal(weights, target, 1)


def test_newtonsmethod():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.newtonsmethod(X, y, gradient(),
                                           hessian())
    target = [0.47, 0.84]
    np.testing.assert_array_almost_equal(weights, target, 1)


def test_newtonsmethod_alpha():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.newtonsmethod(X, y, gradient(),
                                           hessian(), alpha=0.001)
    target = [0.3, 0.5]
    np.testing.assert_array_almost_equal(weights, target, 1)


def test_newtonsmethod_lowiterations():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.newtonsmethod(X, y, gradient(),
                                           hessian(), iterations=2)
    target = [0, 0]
    np.testing.assert_array_almost_equal(weights, target, 1)


def test_newtonsmethod_initialweights():
    x, y = data.continuous_data_complicated()
    X = np.column_stack((np.ones(np.shape(x)[0]), x))
    weights = descentmethods.newtonsmethod(X, y, gradient(),
                                           hessian(),
                                           initial_weights=[.2, .2])
    target = [0.47, 0.84]
    np.testing.assert_array_almost_equal(weights, target, 1)
