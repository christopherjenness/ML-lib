import pytest
import numpy as np

@pytest.fixture
def continuous_data():
    x = np.array(range(5))
    y = np.array(range(5))
    return x, y

@pytest.fixture
def continuous_data_complicated():
    x = np.array(range(10))
    y = np.array(range(5) + [6, 6.1, 6.2, 7, 7.1])
    return x, y

@pytest.fixture
def categorical_data():
    x = np.array(range(6))
    y = np.array([0, 0, 0, 1, 1, 1])
    return x, y

@pytest.fixture
def tall_matrix_data():
    x = np.array([[1, 2, 3],
                    [1.1, 2.05, 3],
                    [0.99, 2, 3],
                    [0.98, 2.1, 3]])
    return x

@pytest.fixture
def categorical_2Dmatrix_data():
    x = np.array([[1, 2],
                  [3, 3],
                  [5, 2],
                  [1, 4],
                  [9, 6],
                  [8, 8]])
    y = np.array([0, 0, 0, 0, 1, 1])
    return x, y

@pytest.fixture
def categorical_2Dmatrix_bernoulli_data():
    x = np.array([[0, 0],
                  [0, 0],
                  [1, 1],
                  [0, 0],
                  [1, 1],
                  [1, 1]])
    y = np.array([0, 0, 0, 0, 1, 1])
    return x, y
