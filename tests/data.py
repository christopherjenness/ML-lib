import pytest
import numpy as np

@pytest.fixture
def continuous_data():
    x = np.array(range(5))
    y = np.array(range(5))
    return x, y

@pytest.fixture
def categorical_data():
    x = np.array(range(6))
    y = np.array([0, 0, 0, 1, 1, 1])
    return x, y
