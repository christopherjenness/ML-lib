import ML.regression as regression
import data
import numpy as np

def test_linear_regression():
    x, y = data.continuous_data()
    lr = regression.LinearRegression()
    lr.fit(x, y)
    predicted_y = lr.predict(x)
    np.testing.assert_array_almost_equal(predicted_y, y)

def test_linear_regression_descent():
    x, y = data.continuous_data()
    lr = regression.LinearRegression()
    lr.fit(x, y, gradient=True)
    predicted_y = lr.predict(x)
    np.testing.assert_array_almost_equal(predicted_y, y)

def test_linear_regression_regularized():
    x, y = data.continuous_data()
    lr = regression.LinearRegression()
    lr.fit(x, y, reg_parameter=0.1)
    predicted_y = lr.predict(x)
    target_y = np.array([0.018688,  1.009157,  1.999626,  2.990095,
                          3.980564])
    np.testing.assert_array_almost_equal(predicted_y, target_y)

def test_linear_regression_descent_regularized():
    x, y = data.continuous_data()
    lr = regression.LinearRegression()
    lr.fit(x, y, gradient=True, reg_parameter=0.1)
    predicted_y = lr.predict(x)
    target_y = np.array([0.018688,  1.009157,  1.999626,  2.990095,
                          3.980564])
    np.testing.assert_array_almost_equal(predicted_y, target_y)

def test_logistic_regression():
    x, y = data.categorical_data()
    lr = regression.LogisticRegression()
    lr.fit(x, y)
    predicted_y = lr.predict(x)
    np.testing.assert_array_almost_equal(predicted_y, y)

def test_logistic_regression_regularized():
    x, y = data.categorical_data()
    lr = regression.LogisticRegression()
    lr.fit(x, y, reg_parameter=0.1)
    predicted_y = lr.predict(x)
    np.testing.assert_array_almost_equal(predicted_y, y)
