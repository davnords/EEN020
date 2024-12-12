import numpy as np

def make_homogenous(x):
    return np.vstack((x.T, np.ones(x.shape[0])))