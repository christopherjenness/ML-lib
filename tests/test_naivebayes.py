import ML.naivebayes as naivebayes
import data
import numpy as np


def test_gaussian_naive_bayes():
    X, y = data.categorical_2Dmatrix_data()
    nb = naivebayes.GaussianNaiveBayes()
    nb.fit(X, y)
    for index, row in enumerate(X):
        predicted_y = nb.predict(row)
        assert predicted_y == y[index]

def test_gaussian_naive_bayes():
    X, y = data.categorical_2Dmatrix_data()
    nb = naivebayes.GaussianNaiveBayes()
    nb.fit(X, y)
    y_probabilities = nb.predict(X[0], probabilities=True)
    assert y_probabilities[y[0]] == 1.0

def test_bernoulli_naive_bayes():
    X, y = data.categorical_2Dmatrix_data()
    nb = naivebayes.BernoulliNaiveBayes()
    nb.fit(X, y)
    for index, row in enumerate(X):
        predicted_y = nb.predict(row)
        assert predicted_y == y[index]

def test_bernoulli_naive_bayes():
    X, y = data.categorical_2Dmatrix_bernoulli_data()
    nb = naivebayes.BernoulliNaiveBayes()
    nb.fit(X, y)
    y_probabilities = nb.predict(X[0], probabilities=True)
    assert y_probabilities[y[0]] == 1.0
