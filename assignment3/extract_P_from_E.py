import numpy as np
from enforce_essential import enforce_essential

def extract_P_from_E(E):
    enforce_essential(E)
    U, S, Vt = np.linalg.svd(E)

    if np.linalg.det(U@Vt)<0:
        Vt = -Vt
    
    u3 = U[:, 2]
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    u3 = np.reshape(u3, (3,1))
    return [np.hstack((U@W@Vt, u3)), np.hstack((U@W@Vt, -u3)), np.hstack((U@W.T@Vt, u3)), np.hstack((U@W.T@Vt, -u3))]