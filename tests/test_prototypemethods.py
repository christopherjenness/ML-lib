import ML.prototypemethods as prototypemthods
import data
import numpy as np


def test_knearestneighbor_regression():
    X, y = data.categorical_2Dmatrix_data_big()
    knn = prototypemthods.KNearestNeighbor()
    knn.fit(X, y)
    prediction = knn.predict(X[0])
    assert prediction == y[0]


def test_knearestneighbor_classification():
    X, y = data.categorical_2Dmatrix_data_big()
    knn = prototypemthods.KNearestNeighbor()
    knn.fit(X, y)
    prediction = knn.predict(X[0], model='classification')
    assert prediction == y[0]


def test_KMeans():
    X, y = data.categorical_2Dmatrix_data_big()
    km = prototypemthods.KMeans()
    km.fit(X)
    assignments = km.sample_assignments
    reversed_assignments = (assignments - 1) * -1
    assert np.array_equal(assignments, y) or \
        np.array_equal(reversed_assignments, y)

def test_KMeans_prediction():
    X, y = data.categorical_2Dmatrix_data_big()
    km = prototypemthods.KMeans()
    km.fit(X)
    assignments = km.sample_assignments
    reversed_assignments = (assignments - 1) * -1
    prediction = km.predict(X[0])
    np.testing.assert_array_almost_equal(prediction, [2.22, 2.5])


def test_KMediods():
    X, y = data.categorical_2Dmatrix_data_big()
    km = prototypemthods.KMediods()
    km.fit(X)
    assignments = km.sample_assignments
    reversed_assignments = (assignments - 1) * -1
    assert np.array_equal(assignments, y) or \
        np.array_equal(reversed_assignments, y)


def test_KMediods_prediction():
    X, y = data.categorical_2Dmatrix_data_big()
    km = prototypemthods.KMediods()
    km.fit(X)
    assignments = km.sample_assignments
    reversed_assignments = (assignments - 1) * -1
    prediction = km.predict(X[0])
    np.testing.assert_array_almost_equal(prediction, [3.0, 3.0])


def test_LVQ():
    X, y = data.categorical_2Dmatrix_data_big()
    lvq = prototypemthods.LearningVectorQuantization()
    lvq.fit(X, y, n_prototypes=3)
    assert [0, 1] == sorted(lvq.prototypes.keys())
    assert (3, 2) == lvq.prototypes[0].shape
    assert (3, 2) == lvq.prototypes[1].shape


def test_LVQ_prediction():
    X, y = data.categorical_2Dmatrix_data_big()
    lvq = prototypemthods.LearningVectorQuantization()
    lvq.fit(X, y, n_prototypes=3)
    prediction = lvq.predict(X[0])
    assert prediction == y[0]
    prediction = lvq.predict(X[-1])
    assert prediction == y[-1]


def test_DANN():
    X, y = data.categorical_2Dmatrix_data_big()
    dann = prototypemthods.DANN()
    dann.fit(X, y)
    assert dann.learned


def test_DANN_prediction():
    X, y = data.categorical_2Dmatrix_data_big()
    dann = prototypemthods.DANN()
    dann.fit(X, y, neighborhood_size=3)
    prediction = dann.predict(X[0])
    assert prediction == y[0]
    prediction = dann.predict(X[-1])
    assert prediction == y[-1]
