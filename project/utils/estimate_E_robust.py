import numpy as np
from .estimate_F_DLT import estimate_F_DLT
from .enforce_essential import enforce_essential
from .compute_epipolar_errors import compute_epipolar_errors
from tqdm import tqdm

def estimate_E_robust(x1,x2,eps, iterations=100):
    """
    I am assuming normalized x1 and x2 (calibrated)
    """
    best_inliers = np.array([False for _ in range(x1.shape[-1])])
    best_E = None
    for i in tqdm(range(iterations), desc="RANSAC iterations"):
        randind = np.random.choice(x1.shape[-1], size=8, replace=False)
        E = enforce_essential(estimate_F_DLT(x1[:,randind],x2[:,randind]))
        
        e1 = np.array(compute_epipolar_errors(E, x1, x2, plot=False))**2
        e2 = np.array(compute_epipolar_errors(E.T, x2, x1, plot=False))**2

        inliers = (1/2)*(e1+e2) < eps**2

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_E = E

    print(f"After running RANSAC for {iterations} iterations the best solution has {best_inliers.sum()} inliers of {x1.shape[-1]} points. I.e. c. {best_inliers.sum()/x1.shape[-1]*100:.2f}%")
    return best_E, best_inliers