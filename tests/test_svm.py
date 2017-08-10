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
