import ML.kernelmethods as kernelmethods
import data
import numpy as np


def test_kernelmethods_fit():
    x, y = data.continuous_data_complicated()
    km = kernelmethods.KernelMethods()
    km.fit(x, y)
    assert km.learned


def test_epanechnikovkernel_close():
    target = kernelmethods.KernelMethods.epanechnikovkernel(0, 0.1, 0.5)
    np.testing.assert_almost_equal(target, 0.72)


def test_epanechnikovkernel_distant():
    target = kernelmethods.KernelMethods.epanechnikovkernel(0, 100, 0.5)
    assert target == 0


def test_epanechnikovkernel_same():
    target = kernelmethods.KernelMethods.epanechnikovkernel(0, 0, 0.5)
    assert target == 0.75


def test_tricubekernel_close():
    target = kernelmethods.KernelMethods.tricubekernel(0, 0.1, 0.5)
    np.testing.assert_almost_equal(target, 0.976, 2)


def test_tricubekernel_distant():
    target = kernelmethods.KernelMethods.tricubekernel(0, 100, 0.5)
    assert target == 0


def test_tricubekernel_same():
    target = kernelmethods.KernelMethods.tricubekernel(0, 0, 0.5)
    assert target == 1


def test_gaussiankernel_close():
    target = kernelmethods.KernelMethods.gaussiankernel(0, 0.1, 0.5)
    np.testing.assert_almost_equal(target, 0.998, 2)


def test_gaussiankernel_distant():
    target = kernelmethods.KernelMethods.gaussiankernel(0, 100, 0.5)
    assert target == 0


def test_gaussiankernel_same():
    target = kernelmethods.KernelMethods.gaussiankernel(0, 0, 0.5)
    assert target == 1


def test_nadarayaaverage():
    x, y = data.continuous_data_complicated()
    km = kernelmethods.KernelMethods()
    km.fit(x, y)
    predict = km.nadarayaaverage(5.5, km.gaussiankernel, 0.01)
    np.testing.assert_almost_equal(predict, 4.237, 2)


def test_locallinearregression():
    x, y = data.continuous_data_complicated()
    km = kernelmethods.KernelMethods()
    km.fit(x, y)
    predict = km.locallinearregression(5.5, km.gaussiankernel, 0.1)
    np.testing.assert_almost_equal(predict, 5.092, 2)


def test_kerneldensityestimate():
    x, y = data.continuous_data_complicated()
    km = kernelmethods.KernelMethods()
    km.fit(x, y)
    kde = km.kerneldensityestimate([5], 1)
    np.testing.assert_almost_equal(kde, 0)


def test_kerneldensitypredict():
    x, y = data.categorical_2Dmatrix_bernoulli_data()
    km = kernelmethods.KernelMethods()
    km.fit(x, y)
    predict = km.kerneldensitypredict([1, 2], 0.8)
    assert predict == 0
