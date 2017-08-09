import ML.gaussianmixture as gaussianmixture
import data
import numpy as np


def test_GM_init():
    GM = gaussianmixture.GaussianMixture()
    GM_five = gaussianmixture.GaussianMixture(c=5)
    assert GM.c == 2
    assert GM_five.c == 5


def test_GM_fit():
    X, y = data.categorical_2Dmatrix_data_big()
    np.random.seed(10)
    GM = gaussianmixture.GaussianMixture()
    GM.fit(X)
    assert GM.learned
    np.testing.assert_array_almost_equal(GM.mus[0],
                                         [2.2, 2.5],
                                         decimal=1)


def test_GM_fit_lowiterations():
    X, y = data.categorical_2Dmatrix_data_big()
    np.random.seed(0)
    GM = gaussianmixture.GaussianMixture()
    GM.fit(X, iterations=1)
    assert GM.learned
    np.testing.assert_array_almost_equal(GM.mus[0],
                                         [2.22, 2.5],
                                         decimal=1)


def test_GM_predict():
    X, y = data.categorical_2Dmatrix_data_big()
    np.random.seed(0)
    GM = gaussianmixture.GaussianMixture()
    GM.fit(X)
    pred_one = GM.predict(X[0])
    pred_two = GM.predict(X[-1])
    assert pred_one == y[0]
    assert pred_two == y[-1]


def test_GM_predict_probs():
    X, y = data.categorical_2Dmatrix_data_big()
    np.random.seed(0)
    GM = gaussianmixture.GaussianMixture()
    GM.fit(X)
    max_class, class_probs = GM.predict(X[0], probs=True)
    assert max_class == 0
    assert np.isclose(class_probs[0], 1)
    assert np.isclose(class_probs[1], 0)
