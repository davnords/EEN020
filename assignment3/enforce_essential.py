import numpy as np
def enforce_essential(E):
    U, S, Vt = np.linalg.svd(E)
    assert np.allclose(S, np.array([1, 1, 0])), 'Essential matrix is not valid!'