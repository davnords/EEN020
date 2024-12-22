from .estimate_E_robust import estimate_E_robust
from .extract_P_from_E import extract_P_from_E
import numpy as np

def parallell_RANSAC(x1, x2, eps, iterations=1000):
    E, inliers = estimate_E_robust(x1, x2, eps, iterations=iterations)

    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2s = np.array(extract_P_from_E(E))
    indices = np.where(np.array([P2[2, :3]@np.array([0, 0, 1]).T for P2 in P2s]) > 0)[0]
    
    
    P2s = P2s[indices]
    # Bad idea because there are two valid camera matrices
    P2 = P2s[0]
    return P2