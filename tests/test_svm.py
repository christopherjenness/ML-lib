import ML.svm as svm
import data
import numpy as np


def test_SupportVectorMachine():
    X, y = data.categorical_2Dmatrix_data_big()
    y = (y * 2) - 1
    SVM = svm.SupportVectorMachine()
    SVM.fit(X, y)
    prediction = SVM.predict(X[0])
    assert prediction == y[0]
    prediction = SVM.predict(X[-1])
    assert prediction == y[-1]


def test_SupportVectorMachine_polynomial_kernel():
    X, y = data.categorical_2Dmatrix_data_big()
    y = (y * 2) - 1
    SVM = svm.SupportVectorMachine(kernel=svm.polynomial_kernel)
    SVM.fit(X, y)
    prediction = SVM.predict(X[0])
    assert prediction == y[0]
    prediction = SVM.predict(X[-1])
    assert prediction == y[-1]


def test_Perceptron():
    X, y = data.categorical_2Dmatrix_data_big()
    y = (y * 2) - 1
    perceptron = svm.Perceptron()
    perceptron.fit(X, y)
    predictions = perceptron.predict(X)
    assert (predictions == y).sum() > 7


def test_linear_kernel():
    x1 = [1, 2, 3]
    x2 = [2, 2, 3]
    kernel = svm.linear_kernel()
    assert kernel(x1, x2) == 15


def test_polynomial_kernel():
    x1 = [1, 2, 3]
    x2 = [2, 2, 3]
    kernel = svm.polynomial_kernel(1, 1)
    assert kernel(x1, x2) == 16
    kernel = svm.polynomial_kernel(2, 2)
    assert kernel(x1, x2) == 289


def test_rbf_kernel():
    x1 = np.array([1, 2, 3])
    x2 = np.array([2, 2, 3])
    kernel = svm.rbf_kernel(1)
    np.testing.assert_almost_equal(kernel(x1, x2), 0.36787, 2)
    kernel = svm.rbf_kernel(2)
    np.testing.assert_almost_equal(kernel(x1, x2), 0.13533, 2)
