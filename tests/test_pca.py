import ML.pca as pca
import data
import numpy as np


def test_PCA():
    X = data.tall_matrix_data()
    pc = pca.PCA()
    pc.fit(X)
    target_x = [3.26739191, 1.52451635, 1]
    np.testing.assert_array_almost_equal(pc.transformed_X[0, :],
                                         target_x)
