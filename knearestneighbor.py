"""

"""
import numpy as np
from scipy import stats

class KNearestNeighbor(object):
    """
    
    """
    def __init__(self):
        """
        
        """
        self.samples = np.nan
        self.values = np.nan
        self.learned = False
        
    def fit(self, X, y):
        """
        
        """
        self.samples = X
        self.values = y
        return self
        
    def predict(self, x, k=1, model='regression'):
        """
        Note: currenly only works on single vector and not matrix
        """
        distances = np.array([])
        for row in range(np.shape(self.samples)[0]):
            #add distance from x to sample row to distances vector
            distances = np.append(distances, np.linalg.norm(x - self.samples[row, :]))
            nearestneighbors = distances.argsort()[-k:]
        if model == 'regression':
            prediction = self.values[nearestneighbors].mean()
        if model == 'classification':
            prediction = stats.mode(self.values[nearestneighbors]).mode
        return prediction