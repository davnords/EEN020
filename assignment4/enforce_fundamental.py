import numpy as np

def enforce_fundamental(F_approx):
    det = np.linalg.det(F_approx)
    assert np.allclose(det, np.zeros((1))), 'Determinant is not zero!'
    