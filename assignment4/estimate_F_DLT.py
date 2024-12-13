import numpy as np

def estimate_F_DLT(x1s, x2s):
    """
    Estimates the fundamental matrix using the Direct Linear Transform (DLT) method.
    
    Parameters:
        x1s: np.ndarray of shape (3, n) - Homogeneous coordinates of points in image 1.
        x2s: np.ndarray of shape (3, n) - Homogeneous coordinates of points in image 2.
        
    Returns:
        F: np.ndarray of shape (3, 3) - The estimated fundamental matrix.
    """
    n = x1s.shape[-1]
    assert n == x2s.shape[-1], "Must be the same number of points"
    
    # Construct the M matrix using broadcasting and stacking
    x1, y1, w1 = x1s
    x2, y2, w2 = x2s
    
    M = np.stack([
        x2 * x1, x2 * y1, x2 * w1,
        y2 * x1, y2 * y1, y2 * w1,
        w2 * x1, w2 * y1, w2 * w1
    ], axis=1)
    
    # Perform SVD on M
    _, S, Vt = np.linalg.svd(M)
    F = Vt[-1].reshape(3, 3)  # Extract the last row of Vt and reshape to 3x3
    
    # Enforce rank-2 constraint on F by setting the smallest singular value to zero
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0  # Set the smallest singular value to zero
    F = U @ np.diag(S) @ Vt
    
    return F
