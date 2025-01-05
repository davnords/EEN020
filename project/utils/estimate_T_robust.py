from tqdm import tqdm
import numpy as np


def estimate_T_robust(x, X, R, eps, iterations=10):
    """
    Estimate the translation vector robustly by running RANSAC and with 2 point correspondences in DLT,
    assuming the rotation matrix is known.
    
    Arguments:
    x -- 2D points in the image (3xN)
    X -- 3D points in the world (4xN)
    R -- Known 3x3 rotation matrix
    
    Returns:
    P -- The full projection matrix (3x4)
    """
    best_inliers = np.array([False for _ in range(x.shape[-1])])
    best_P = None

    for i in tqdm(range(iterations), desc="RANSAC iterations"):
        randind = np.random.choice(x.shape[1], size=2, replace=False)
        P = estimate_translation_DLT_SVD(x[:, randind], X[:, randind], R)
        projected_points = P@X
        projected_points /= projected_points[2, :]
        errors = np.linalg.norm(x[:2, :] - projected_points[:2, :], axis=0)**2 
        inliers = errors < eps

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_P = P
    print(f"After running RANSAC for {iterations} iterations the best solution has {best_inliers.sum()} inliers of {x.shape[-1]} points. I.e. c. {best_inliers.sum()/x.shape[-1]*100:.2f}%")
    return best_P
    
        
        
def estimate_translation_DLT_SVD(x, X, R, printSV=False):
    """Estimate the camera translation vector given known rotation matrix using 2 or more points.
    
    Arguments:
    x -- 2D points in the image (3xN)
    X -- 3D points in the world (4xN)
    R -- Known 3x3 rotation matrix
    """
    if x.shape[0] != 3 or X.shape[0] != 4:
        raise ValueError('Input must be a 3xN matrix')
    if x.shape[1] != X.shape[1]:
        raise ValueError('The number of points in x and X must be the same')
    if R.shape != (3, 3):
        raise ValueError('R must be a 3x3 rotation matrix')
    
    N = x.shape[1]
    # We only need 2 points minimum (4 equations) to solve for 3 unknowns
    if N < 2:
        raise ValueError('At least 2 points are required')
    
    # Create the measurement matrix M
    M = np.zeros((2*N, 3))
    b = np.zeros(2*N)
    
    for i in range(N):
        # Get rotated 3D point
        X_rotated = R @ X[0:3, i]
        
        # First equation: x = (R*X + t)_1 / (R*X + t)_3
        M[2*i, :] = [1, 0, -x[0, i]]
        b[2*i] = x[0, i]*X_rotated[2] - X_rotated[0]
        
        # Second equation: y = (R*X + t)_2 / (R*X + t)_3
        M[2*i+1, :] = [0, 1, -x[1, i]]
        b[2*i+1] = x[1, i]*X_rotated[2] - X_rotated[1]
    
    # Solve the system usig least squares
    t = np.linalg.lstsq(M, b, rcond=None)[0]
    
    # Construct the full camera matrix
    P = np.column_stack([R, t.reshape(3, 1)])
    

    return P