import numpy as np

def estimate_camera_DLT(x, X):
    """Estimate the camera matrix using the DLT algorithm.
    Arguments:
    x -- 2D points in the image (3xN)
    X -- 3D points in the world (4xN)
    """
    if x.shape[0] != 3 or X.shape[0] != 4:
        raise ValueError('Input must be a 3xN matrix')
    if x.shape[1] != X.shape[1]:
        raise ValueError('The number of points in x and X must be the same')
    N = x.shape[1]
    M = np.zeros((2*N, 12))
    for i in range(N):
        M[2*i, 0:4] = X[:, i]
        M[2*i, 8:12] = -x[0, i]*X[:, i]
        M[2*i+1, 4:8] = X[:, i]
        M[2*i+1, 8:12] = -x[1, i]*X[:, i]
    _, S, Vt = np.linalg.svd(M)
    print(f"The smallest singular value is: {S[-1]:.4f}")
    print(f"The value of ||Mv|| is: {np.linalg.norm(np.dot(M, Vt[-1])):.4f}")
    P = Vt[-1].reshape(3, 4)
    return P