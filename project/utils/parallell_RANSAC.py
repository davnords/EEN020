from .estimate_E_robust import estimate_E_robust
from .extract_P_from_E import extract_P_from_E
import numpy as np

def parallell_RANSAC(x1, x2, eps, iterations=1000):
    E, inliers = estimate_E_robust(x1, x2, eps, iterations=iterations)

    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2s = extract_P_from_E(E)