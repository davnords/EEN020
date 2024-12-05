import numpy as np
def enforce_essential(E):
    U, S, Vt = np.linalg.svd(E)
    S = np.diag([1, 1, 0])
    E = U@S@Vt

    return E