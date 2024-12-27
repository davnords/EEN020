from .estimate_E_robust import estimate_E_robust
from .extract_P_from_E import extract_P_from_E
from .triangulate_3D_point_DLT import triangulate_3D_point_DLT
import numpy as np

def parallell_RANSAC(x1, x2, eps, iterations=1000):
    E, inliers = estimate_E_robust(x1, x2, eps, iterations=iterations)

    P1 = np.hstack((np.eye(3), np.zeros((3,1))))
    P2s = np.array(extract_P_from_E(E))

    chierality_camera = 0
    for i, P2 in enumerate(P2s):
        Xi, _ = triangulate_3D_point_DLT(x1[:2, 0], x2[:2, 0], P1, P2)
        X_h = np.hstack((Xi, [1]))  # Homogeneous coordinates
        if P1[2, :] @ X_h > 0 and P2[2, :] @ X_h > 0:
            chierality_camera = i
    return P2s[chierality_camera]