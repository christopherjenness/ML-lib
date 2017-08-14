import ML.treemethods as treemethods
import data
import numpy as np


def test_RegressionTree():
    tree = treemethods.RegressionTree()
    X, y = data.categorical_2Dmatrix_data_big()
    tree.fit(X, y, 3)
    assert tree.predict(X[0]) == y[0]
    assert tree.predict(X[-1]) == y[-1]


def test_ClassificationTree():
    tree = treemethods.ClassificationTree()
    X, y = data.categorical_2Dmatrix_data_big()
    tree.fit(X, y, 3)
    assert tree.predict(X[0]) == y[0]
    assert tree.predict(X[-1]) == y[-1]


def test_PrimRegression():
    tree = treemethods.PrimRegression()
    X, y = data.categorical_2Dmatrix_data_big()
    tree.fit(X, y, 1)
    assert tree.predict(X[1]) == y[1]
    assert np.isclose(tree.predict(X[-1]), 0.6666, 1)


def test_DiscreteAdaBoost():
    tree = treemethods.DiscreteAdaBoost()
    X, y = data.categorical_2Dmatrix_data_big()
    tree.fit(X, y, 3)
    assert tree.predict(X[0]) == y[0]
    assert tree.predict(X[-1]) == y[-1]


def test_GradientBoostingRegression():
    tree = treemethods.GradientBoostingRegression()
    X, y = data.categorical_2Dmatrix_data_big()
    tree.fit(X, y, 3)
    assert np.isclose(tree.predict(X[0]), 0.3976, 1)
    assert np.isclose(tree.predict(X[-1]), y[-1], 1)


def test_RandomForestRegression():
    tree = treemethods.RandomForestRegression()
    X, y = data.categorical_2Dmatrix_data_big()
    tree.fit(X, y, 3)
    assert tree.predict(X[0]) == y[0]
    assert tree.predict(X[-1]) == y[-1]
