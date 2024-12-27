from tqdm import tqdm
import numpy as np
from .general import make_homogenous

def estimate_T_robust(xs, Xs, R, eps, iterations=100):
    best_inliers = np.array([False for _ in range(xs.shape[-1])])
    best_T = None
    Xs = make_homogenous(Xs.T)

    for i in tqdm(range(iterations), desc="RANSAC iterations"):
        # Randomly select 2 correspondences
        randind = np.random.choice(xs.shape[1], size=2, replace=False)
        x, X = xs[:, randind], Xs[:, randind]
        
        # Construct the M matrix based on the 2D-3D correspondences
        M = np.zeros((6, 4))
        for j in range(2):  # Two correspondences
            Xj = X[:, j]
            xj = x[:, j]

            M[3*j + 0] = [Xj[-1], 0, 0, R[0, :]@Xj[:3]-xj[0]]
            M[3*j + 1] = [0, Xj[-1], 0, R[1, :]@Xj[:3]-xj[1]]
            M[3*j + 2] = [0, 0, Xj[-1], R[2, :]@Xj[:3]-xj[2]]
        
        # Solve the system using SVD to find the null space
        _, _, Vt = np.linalg.svd(M)
        T = Vt[-1, :3].reshape(-1, 1)  # The translation is the last column of Vt
        
        # Trying another way to run the DLT
        _, T = estimate_camera_center_DLT(x, X, R, printSV=False)

        # Use reprojection error to check for inliers
        projected_points = R @ Xs[:3, :] + T  # 3 x N
        projected_points /= projected_points[2, :]  # Normalize to homogeneous (2D)
        errors = np.linalg.norm(xs[:2, :] - projected_points[:2, :], axis=0)  # Euclidean distance


        inliers = errors < eps
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_T = T

    print(f"After running RANSAC for {iterations} iterations the best solution has {best_inliers.sum()} inliers of {xs.shape[-1]} points. I.e. c. {best_inliers.sum()/xs.shape[-1]*100:.2f}%")
    return best_T, best_inliers


def estimate_camera_center_DLT(x, X, R, printSV=False):
    """Estimate the camera center using the DLT algorithm with known rotation.
    Only requires 2 point correspondences since rotation is known.
    
    Arguments:
    x -- 2D points in the image (3xN), N >= 2
    X -- 3D points in the world (4xN), N >= 2
    R -- Known 3x3 rotation matrix
    """
    if x.shape[0] != 3 or X.shape[0] != 4:
        raise ValueError('Input must be a 3xN matrix')
    if x.shape[1] < 2:
        raise ValueError('At least 2 point correspondences are required')
    if x.shape[1] != X.shape[1]:
        raise ValueError('The number of points in x and X must be the same')
    if R.shape != (3, 3):
        raise ValueError('R must be a 3x3 rotation matrix')
    
    # We only need 2 points, but we'll use all provided points for better stability
    N = x.shape[1]
    M = np.zeros((2*N, 3))

    for i in range(N):
        # For each point correspondence:
        # x = K[R|t]X = K(RX + t)
        # Since K and R are known, we can solve for t
        M[2*i, :] = x[0, i] * R[2, :] - R[0, :]
        M[2*i+1, :] = x[1, i] * R[2, :] - R[1, :]
    
    # Solve for translation using SVD
    U, S, Vt = np.linalg.svd(M)
    if printSV:
        print(f"The smallest singular value is: {S[-1]:.4f}")
        print(f"The value of ||Mv|| is: {np.linalg.norm(np.dot(M, Vt[-1])):.4f}")
    
    # Get translation vector from the last right singular vector
    t = Vt[-1]
    
    # Construct final camera matrix
    P = np.hstack((R, t.reshape(3, 1)))
    
    # Check if points are in front of the camera
    x_projected = P @ X
    depths = x_projected[2, :]
    if np.any(depths <= 0):
        t = -t  # Negate the translation to flip the camera direction
        P = np.hstack((R, t.reshape(3, 1)))
    
    return P, t.reshape(-1, 1)